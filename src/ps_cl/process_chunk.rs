use std::mem::size_of;
use std::io::{Cursor, Write};
use std::collections::HashMap;
use std::{thread, thread::JoinHandle};
use std::slice::{from_raw_parts, from_raw_parts_mut};
use itertools::izip;
use anyhow::{Error, Result};
use tracing::{debug, info, error};
use crossbeam_channel::{Sender, Receiver};
use ocl::{ProQue, Buffer, MemFlags, Event};
use super::{EventChunk, EV_BUFFER_N_JOBS, EV_BUFFER_N_BUCKET, EV_BUFFER_BUCKET_SIZE, OBS_MAP_RESOLUTION};
use super::{gather_ls_ps::GatherLSPS, gather_ps_fcn_cnn_ps::GatherPSFCNCNNPS};

const EV_LEN: usize = size_of::<u16>() + size_of::<u16>() + size_of::<u8>() + size_of::<u32>();
const OBS_MAP_N_PIXEL: usize = OBS_MAP_RESOLUTION * OBS_MAP_RESOLUTION;

// Must use a separated function to annotate lifetime of the returned curser
fn buffer_map_write(buffer_map: &mut ocl::MemMap<u8>) -> Result<Cursor<&mut [u8]>> {
  let buffer_slice = unsafe { from_raw_parts_mut(buffer_map.as_mut_ptr(), buffer_map.len()) };
  Ok(Cursor::new(buffer_slice))
}

struct SendPSFCNCNNPS {
  record_n_bin: usize,
  per_n_bin: u32,
  time_per_bin: u32,
  pixel_buffer: Option<Buffer<u8>>,
  total_i_bin: usize,
}

impl SendPSFCNCNNPS {
  fn new(
      method: &str,
      config_ps: &HashMap<String, String>,
      pixel_buffer_len: usize,
      proque: &ProQue) -> Result<Option<Self>> {
    let per_n_bin = config_ps[&format!("{method}_per_n_bin")].parse::<u32>()?;
    if per_n_bin == 0 {
      return Ok(None);
    }
    let record_n_bin = config_ps["record_n_bin"].parse::<usize>()?;
    let record_time = config_ps["record_time"].parse::<u32>()?;
    let time_per_bin = record_time / record_n_bin as u32;
    let pixel_buffer = Some(Buffer::builder()
        .queue(proque.queue().to_owned())
        .flags(MemFlags::READ_ONLY)
        .len(pixel_buffer_len)
        .fill_val(0)
        .build()?);
    Ok(Some(SendPSFCNCNNPS{
      record_n_bin,
      per_n_bin,
      time_per_bin,
      pixel_buffer,
      total_i_bin: 0,
    }))
  }

  fn send(
      &mut self,
      timestamp_max: u32,
      pixel_buffer: &Buffer<u8>,
      event_execution: &Event,
      channel: &(Receiver<Buffer<u8>>, Sender<Buffer<u8>>)) -> Result<()> {
    // Considering u32 overflow
    let ps_fcn_delta_timestamp = timestamp_max - self.total_i_bin as u32 * self.time_per_bin;
    let ps_fcn_per_timestamp = self.per_n_bin * self.time_per_bin;
    if ps_fcn_delta_timestamp > ps_fcn_per_timestamp {
      self.total_i_bin += self.per_n_bin as usize;
      if self.total_i_bin >= self.record_n_bin {
        let self_pixel_buffer = self.pixel_buffer.take().ok_or(()).or_else(|_| {
          channel.0.recv()
        })?;
        let mut event_read = Event::empty();
        pixel_buffer.copy(&self_pixel_buffer, None, None)
          .ewait(event_execution)
          .enew(&mut event_read)
          .enq()?;
        event_read.wait_for()?;
        channel.1.send(self_pixel_buffer)?;
      }
    }
    Ok(())
  }
}

fn process_chunk(
    n_row: u16,
    n_col: u16,
    i_job: u16,
    config_ps: &HashMap<String, String>,
    ev_chunk_receiver: Receiver<Box<EventChunk>>,
    ls_ps_channel: Option<(Receiver<Buffer<u8>>, Sender<(u16, u32, Buffer<u8>)>)>,
    ps_fcn_channel: Option<(Receiver<Buffer<u8>>, Sender<Buffer<u8>>)>,
    cnn_ps_channel: Option<(Receiver<Buffer<u8>>, Sender<Buffer<u8>>)>,
    proque: &ProQue) -> Result<JoinHandle<()>> {
  let n_pixels_per_job = n_row as usize * n_col as usize / EV_BUFFER_N_JOBS;
  let record_n_bin = config_ps["record_n_bin"].parse::<usize>()?;
  let pixel_buffer_len = n_pixels_per_job * (size_of::<u8>() + size_of::<u32>() +
    if ls_ps_channel.is_some() { size_of::<f32>() * (1 + 6) } else { 0 } +
    if ps_fcn_channel.is_some() { size_of::<f32>() * 3 * record_n_bin } else { 0 } +
    if cnn_ps_channel.is_some() { (size_of::<f32>() * 3 + size_of::<u32>()) * OBS_MAP_N_PIXEL } else { 0 });
  let mut send_ps_fcn = SendPSFCNCNNPS::new("ps_fcn", config_ps, pixel_buffer_len, proque)?;
  let mut send_cnn_ps = SendPSFCNCNNPS::new("cnn_ps", config_ps, pixel_buffer_len, proque)?;
  let ev_chunk_buffer_len =
    size_of::<u16>() * EV_BUFFER_N_BUCKET +
    EV_LEN * EV_BUFFER_BUCKET_SIZE * EV_BUFFER_N_BUCKET;
  let ev_chunk_buffer = Buffer::builder()
    .queue(proque.queue().to_owned())
    .flags(MemFlags::READ_ONLY)
    .len(ev_chunk_buffer_len)
    .build()?;
  let pixel_buffer = Buffer::builder()
    .queue(proque.queue().to_owned())
    .len(pixel_buffer_len)
    .fill_val(0)
    .build()?;
  if let Some((_, ls_ps_response_sender)) = &ls_ps_channel {
    let pixel_buffer_ls_ps = Buffer::builder()
      .queue(proque.queue().to_owned())
      .flags(MemFlags::READ_ONLY)
      .len(pixel_buffer_len)
      .fill_val(0)
      .build()?;
    ls_ps_response_sender.send((i_job, 0, pixel_buffer_ls_ps))?;
  }
  let kernel_process_chunk = proque.kernel_builder("kernel_process_chunk")
    .arg(&ev_chunk_buffer)
    .arg(&pixel_buffer)
    .arg(0u32)
    .arg(0u32)
    .global_work_size(EV_BUFFER_N_BUCKET)
    .build()?;
  Ok(thread::spawn(move || {
    info!("PSCL::spawn_process_chunk: thread started");
    let ret = (move || {
      let mut event_execution = Event::empty();
      // Stop thread if `ev_chunk_sender` is closed
      while let Ok(ev_chunk) = ev_chunk_receiver.recv() {
        let ev_total = ev_chunk.n_ev.iter().fold(0, |b, a| *a as usize + b);
        let ev_capacity = ev_chunk.polarity.len() * ev_chunk.polarity[0].len();
        debug!("Got chunk! Utility: {:.5}", ev_total as f32 / ev_capacity as f32);
        if let (Some(send_ps_fcn), Some(ps_fcn_channel)) = (&mut send_ps_fcn, &ps_fcn_channel) {
          send_ps_fcn.send(ev_chunk.timestamp_max, &pixel_buffer, &event_execution, ps_fcn_channel)?;
        }
        if let (Some(send_cnn_ps), Some(cnn_ps_channel)) = (&mut send_cnn_ps, &cnn_ps_channel) {
          send_cnn_ps.send(ev_chunk.timestamp_max, &pixel_buffer, &event_execution, cnn_ps_channel)?;
        }
        {
          let mut ev_chunk_buffer_map = unsafe { ev_chunk_buffer.map().ewait(&event_execution).write().enq() }?;
          let mut writer = buffer_map_write(&mut ev_chunk_buffer_map)?;
          // Assume host and device have the same byteorder
          unsafe {
            writer.write_all(from_raw_parts(ev_chunk.n_ev.as_ptr() as *const u8,
                                            ev_chunk.n_ev.len() * size_of::<u16>()))?;
          }
          for arr in ev_chunk.i_row {
            unsafe {
              writer.write_all(from_raw_parts(arr.as_ptr() as *const u8, arr.len() * size_of::<u16>()))?;
            }
          }
          for arr in ev_chunk.i_col {
            unsafe {
              writer.write_all(from_raw_parts(arr.as_ptr() as *const u8, arr.len() * size_of::<u16>()))?;
            }
          }
          for arr in ev_chunk.polarity {
            writer.write_all(&arr)?;
          }
          for arr in ev_chunk.timestamp {
            unsafe {
              writer.write_all(from_raw_parts(arr.as_ptr() as *const u8, arr.len() * size_of::<u32>()))?;
            }
          }
          assert_eq!(writer.position().try_into(), Ok(ev_chunk_buffer_len));
        }
        kernel_process_chunk.set_arg(2, ev_chunk.timestamp_begin)?;
        kernel_process_chunk.set_arg(3, ev_chunk.timestamp_end)?;
        event_execution = Event::empty();
        unsafe { kernel_process_chunk.cmd().enew(&mut event_execution).enq()?; }
        if let Some((ls_ps_request_receiver, ls_ps_response_sender)) = &ls_ps_channel {
          if let Ok(pixel_buffer_ls_ps) = ls_ps_request_receiver.try_recv() {
            let mut event_read = Event::empty();
            pixel_buffer.copy(&pixel_buffer_ls_ps, None, None)
              .ewait(&event_execution)
              .enew(&mut event_read)
              .enq()?;
            event_read.wait_for()?;
            ls_ps_response_sender.send((i_job, ev_chunk.timestamp_max, pixel_buffer_ls_ps))?;
          }
        }
      }
      Ok::<_, Error>(String::from("`ev_chunk_sender` closed"))
    })();
    if let Err(e) = ret {
      error!("PSCL::spawn_process_chunk: {e:?}");
      panic!("PSCL::spawn_process_chunk: {e:?}");
    }
    info!("PSCL::spawn_process_chunk: thread exit {ret:?}");
  }))
}

fn request_channel<T>(
    enable: bool,
    ) -> (Option<[Sender<T>; EV_BUFFER_N_JOBS]>, [Option<Receiver<T>>; EV_BUFFER_N_JOBS]) {
  if enable {
    let (s, r): (Vec<_>, Vec<_>) = (0..EV_BUFFER_N_JOBS).map(|_| {
      let (s, r) = crossbeam_channel::bounded(1);
      (s, Some(r))
    }).unzip();
    (Some(s.try_into().expect("EV_BUFFER_N_JOBS")), r.try_into().expect("EV_BUFFER_N_JOBS"))
  } else {
    Default::default()
  }
}

fn response_channel<T>(
    enable: bool,
    ) -> ([Option<Sender<T>>; EV_BUFFER_N_JOBS], Option<[Receiver<T>; EV_BUFFER_N_JOBS]>) {
  if enable {
    let (s, r): (Vec<_>, Vec<_>) = (0..EV_BUFFER_N_JOBS).map(|_| {
      let (s, r) = crossbeam_channel::bounded(1);
      (Some(s), r)
    }).unzip();
    (s.try_into().expect("EV_BUFFER_N_JOBS"), Some(r.try_into().expect("EV_BUFFER_N_JOBS")))
  } else {
    Default::default()
  }
}

pub fn spawn_process_chunk(
    n_row: u16,
    n_col: u16,
    config_ps: &HashMap<String, String>,
    arr_ev_chunk_receiver: [Receiver<Box<EventChunk>>; EV_BUFFER_N_JOBS],
    proque: &ProQue,
    cl_render_buffer: &ocl::core::Mem,
    ) -> Result<(Vec<JoinHandle<()>>, Option<GatherLSPS>, Option<GatherPSFCNCNNPS>, Option<GatherPSFCNCNNPS>)> {
  let enable_ls_ps = matches!(config_ps["show_ls_ps"].as_str(), "gl" | "cv" | "none");
  let enable_ps_fcn = config_ps["ps_fcn_per_n_bin"].parse::<u32>()? > 0;
  let enable_cnn_ps = config_ps["cnn_ps_per_n_bin"].parse::<u32>()? > 0;
  let (arr_ls_ps_request_sender, arr_ls_ps_request_receiver) = request_channel(enable_ls_ps);
  let (ls_ps_response_sender, ls_ps_response_receiver) = if enable_ls_ps {
    let (s, r) = crossbeam_channel::bounded(EV_BUFFER_N_JOBS);
    (Some(s), Some(r))
  } else {
    Default::default()
  };
  let (arr_ps_fcn_request_sender, arr_ps_fcn_request_receiver) = request_channel(enable_ps_fcn);
  let (arr_ps_fcn_response_sender, arr_ps_fcn_response_receiver) = response_channel(enable_ps_fcn);
  let (arr_cnn_ps_request_sender, arr_cnn_ps_request_receiver) = request_channel(enable_cnn_ps);
  let (arr_cnn_ps_response_sender, arr_cnn_ps_response_receiver) = response_channel(enable_cnn_ps);
  let vec_task_process_chunk = izip!(0..EV_BUFFER_N_JOBS,
                                     arr_ev_chunk_receiver,
                                     arr_ls_ps_request_receiver,
                                     arr_ps_fcn_response_sender,
                                     arr_ps_fcn_request_receiver,
                                     arr_cnn_ps_response_sender,
                                     arr_cnn_ps_request_receiver).map(|(i_job,
                                                                        ev_chunk_receiver,
                                                                        ls_ps_request_receiver,
                                                                        ps_fcn_response_sender,
                                                                        ps_fcn_request_receiver,
                                                                        cnn_ps_response_sender,
                                                                        cnn_ps_request_receiver)| {
      let ls_ps_channel = enable_ls_ps.then(||
        (ls_ps_request_receiver.expect("enable_ls_ps"), ls_ps_response_sender.clone().expect("enable_ls_ps")));
      let ps_fcn_channel = enable_ps_fcn.then(||
        (ps_fcn_request_receiver.expect("enable_ps_fcn"), ps_fcn_response_sender.expect("enable_ps_fcn")));
      let cnn_ps_channel = enable_cnn_ps.then(||
        (cnn_ps_request_receiver.expect("enable_cnn_ps"), cnn_ps_response_sender.expect("enable_cnn_ps")));
      process_chunk(n_row,
                    n_col,
                    i_job.try_into()?,
                    config_ps,
                    ev_chunk_receiver,
                    ls_ps_channel,
                    ps_fcn_channel,
                    cnn_ps_channel,
                    &proque)
    }).collect::<Result<Vec<_>>>()?;
  let gather_ls_ps = enable_ls_ps.then(|| {
    GatherLSPS::new(n_row,
                    n_col,
                    &proque,
                    cl_render_buffer,
                    arr_ls_ps_request_sender.expect("enable_ls_ps"),
                    ls_ps_response_receiver.expect("enable_ls_ps"))
  }).transpose()?;
  let gather_ps_fcn = enable_ps_fcn.then(|| {
    GatherPSFCNCNNPS::new("ps_fcn",
                          n_row,
                          n_col,
                          config_ps,
                          &proque,
                          arr_ps_fcn_request_sender.expect("enable_ps_fcn"),
                          arr_ps_fcn_response_receiver.expect("enable_ps_fcn"))
  }).transpose()?;
  let gather_cnn_ps = enable_cnn_ps.then(|| {
    GatherPSFCNCNNPS::new("cnn_ps",
                          n_row,
                          n_col,
                          config_ps,
                          &proque,
                          arr_cnn_ps_request_sender.expect("enable_cnn_ps"),
                          arr_cnn_ps_response_receiver.expect("enable_cnn_ps"))
  }).transpose()?;
  Ok((vec_task_process_chunk, gather_ls_ps, gather_ps_fcn, gather_cnn_ps))
}
