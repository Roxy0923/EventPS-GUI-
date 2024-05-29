use std::io::Write;
use std::boxed::Box;
use std::mem::size_of;
use std::collections::HashMap;
use std::slice::from_raw_parts;
use std::sync::{Arc, Weak, Mutex, atomic::{AtomicU32, AtomicU64, Ordering}};
use cxx::SharedPtr;
use itertools::izip;
use ndarray::prelude::*;
use xz2::write::XzEncoder;
use anyhow::{Result, ensure};
use tracing::{debug, info, warn, error};
use crossbeam_channel::{Sender, Receiver};
use super::*;

/// The light direction start from right, rotating clockwise

#[cxx::bridge(namespace = "EventPS")]
mod ffi {
  extern "Rust" {
    type RustCamera;
    fn cd_callback(
      camera_data_weak: Box<RustCamera>,
      i_job: u16,
      i_row: &[u16],
      i_col: &[u16],
      polarity: &[u8],
      timestamp: &[u32]) -> Box<RustCamera>;
    fn ext_trigger_callback(
      camera_data_weak: Box<RustCamera>,
      timestamp: u32,
      pixel_dead_time: u32) -> Box<RustCamera>;
  }
  unsafe extern "C++" {
    include!("event_ps/include/loader_prophesee.h");
    type Camera;
    fn create_camera(
      camera_data_weak: Box<RustCamera>,
      serial: &str,
      begin_row: u16,
      end_row: u16,
      begin_col: u16,
      end_col: u16,
      bias_fo: i32,
      bias_diff_on: i32,
      bias_diff_off: i32,
      bias_refr: i32) -> SharedPtr<Camera>;
    fn stop_camera(camera: SharedPtr<Camera>);
  }
}

struct PropheseeData {
  cd_counter: AtomicU32,
  ext_trigger_counter: AtomicU32,
  arr_ev_sender: [Sender<Box<EventCD>>; EV_BUFFER_N_JOBS],
  timestamp_begin_end: AtomicU64,
  event_writer: Option<Mutex<(XzEncoder<File>, File)>>,
}

pub struct RustCamera(Weak<PropheseeData>);

pub struct Prophesee {
  camera: SharedPtr<ffi::Camera>,
  camera_data: Arc<PropheseeData>,
}

unsafe impl Send for Prophesee {}

impl Loader for Prophesee {
  fn new(
      config_loader_prophesee: &HashMap<String, String>,
      #[cfg(feature = "display_cv")] _cv_mutex: &'static Mutex<()>,
      )-> Result<(Self, [Receiver<Box<EventChunk>>; EV_BUFFER_N_JOBS])> {
    let begin_row = config_loader_prophesee["begin_row"].parse::<u16>()?;
    let end_row = config_loader_prophesee["end_row"].parse::<u16>()?;
    let begin_col = config_loader_prophesee["begin_col"].parse::<u16>()?;
    let end_col = config_loader_prophesee["end_col"].parse::<u16>()?;
    ensure!(begin_col as usize % EV_BUFFER_N_JOBS == 0, "EV_BUFFER_N_JOBS must be a factor of begin_col");
    ensure!(end_col as usize % EV_BUFFER_N_JOBS == 0, "EV_BUFFER_N_JOBS must be a factor of end_col");
    let save_event = &config_loader_prophesee["save_event"];
    let (vec_ev_sender, vec_ev_receiver) = (0..EV_BUFFER_N_JOBS).map(|_| {
      if save_event.is_empty() {
        crossbeam_channel::bounded(EV_CHANNEL_SIZE)
      } else {
        info!("loader_prophesee: use unbounded ev channel to save events");
        crossbeam_channel::unbounded()
      }
    }).unzip::<_, _, Vec<_>, Vec<_>>();
    let event_writer = new_event_writer(save_event, 0)?.map(|writer| { Mutex::new(writer) });
    let camera_data = Arc::new(PropheseeData {
      cd_counter: AtomicU32::new(0),
      ext_trigger_counter: AtomicU32::new(0),
      arr_ev_sender: vec_ev_sender.try_into().expect("EV_BUFFER_N_JOBS"),
      timestamp_begin_end: AtomicU64::new(0),
      event_writer,
    });
    let vec_ev_chunk_receiver = vec_ev_receiver.into_iter().map(|ev_receiver| {
      let camera_data_weak = Arc::downgrade(&camera_data);
      let update = Box::new(move |ev: &EventCD| {
        // Stop thread if `camera_data` is dropped
        let Some(camera_data) = camera_data_weak.upgrade() else {
          return None;
        };
        if let Some(event_writer) = &camera_data.event_writer {
          let mut events = Array2::default((EV_BUFFER_SIZE, 4));
          for (mut event, i_row, i_col, polarity, timestamp) in
              izip!(events.outer_iter_mut(), &ev.i_row, &ev.i_col, &ev.polarity, &ev.timestamp) {
            event[0] = (*i_row - begin_row) as f32;
            event[1] = (*i_col - begin_col) as f32;
            event[2] = if *polarity > 0 { 1. } else { -1. };
            event[3] = *timestamp as f32 / 1e6;
          }
          let event_writer = &mut event_writer.lock().expect("No task should panic").0;
          if let Err(e) = unsafe {
            event_writer.write_all(from_raw_parts(events.as_ptr() as *const u8, events.len() * size_of::<f32>()))
          } {
            error!("loader_prophesee::update: {e:?}");
            panic!("loader_prophesee::update: {e:?}");
          }
        }
        camera_data.cd_counter.fetch_add(ev.n_ev as u32, Ordering::Relaxed);
        let timestamp_begin_end = camera_data.timestamp_begin_end.load(Ordering::Relaxed);
        Some(((timestamp_begin_end >> 32) as u32, timestamp_begin_end as u32))
      });
      let (_, ev_chunk_receiver) = spawn_split_event(begin_row, end_row, begin_col, end_col, ev_receiver, update);
      ev_chunk_receiver
    }).collect::<Vec<_>>();
    let camera_data_weak = Box::new(RustCamera(Arc::downgrade(&camera_data)));
    let serial = &config_loader_prophesee["serial"];
    let camera = Prophesee {
      camera: ffi::create_camera(camera_data_weak,
                                 serial,
                                 begin_row,
                                 end_row,
                                 begin_col,
                                 end_col,
                                 config_loader_prophesee["bias_fo"].parse::<i32>()?,
                                 config_loader_prophesee["bias_diff_on"].parse::<i32>()?,
                                 config_loader_prophesee["bias_diff_off"].parse::<i32>()?,
                                 config_loader_prophesee["bias_refr"].parse::<i32>()?),
      camera_data,
    };
    Ok((camera, vec_ev_chunk_receiver.try_into().expect("EV_BUFFER_N_JOBS")))
  }

  fn get_cd_counter(&self) -> u32 {
    self.camera_data.cd_counter.load(Ordering::Relaxed)
  }

  fn get_ext_trigger_counter(&self) -> Option<u32> {
    Some(self.camera_data.ext_trigger_counter.load(Ordering::Relaxed))
  }
}

impl Drop for Prophesee {
  fn drop(&mut self) {
    ffi::stop_camera(self.camera.to_owned());
  }
}

fn cd_callback(
    camera_data_weak: Box<RustCamera>,
    i_job: u16,
    i_row: &[u16],
    i_col: &[u16],
    polarity: &[u8],
    timestamp: &[u32]) -> Box<RustCamera> {
  let Some(camera_data) = camera_data_weak.0.upgrade() else {
    warn!("cd_callback is called but camera_data is dropped!");
    return camera_data_weak;
  };
  let ev = Box::new(EventCD::from_slice(i_row, i_col, polarity, timestamp).expect("Size Should Be EV_BUFFER_SIZE"));
  if camera_data.arr_ev_sender[i_job as usize].is_full() {
    warn!("ev channel full!");
  }
  camera_data.arr_ev_sender[i_job as usize].send(ev).expect("Receiver Should Not Panic");
  camera_data_weak
}

fn ext_trigger_callback(camera_data_weak: Box<RustCamera>, timestamp: u32, pixel_dead_time: u32) -> Box<RustCamera> {
  debug!("ext_trigger_callback: pixel_dead_time = {pixel_dead_time}us");
  let Some(camera_data) = camera_data_weak.0.upgrade() else {
    warn!("ext_trigger_callback is called but camera_data is dropped!");
    return camera_data_weak;
  };
  if let Some(event_writer) = &camera_data.event_writer {
    let event_writer = &mut event_writer.lock().expect("No task should panic").1;
    if let Err(e) = event_writer.write_all(&(timestamp as f32 / 1e6).to_ne_bytes()) {
      error!("loader_prophesee::ext_trigger_callback: {e:?}");
      panic!("loader_prophesee::ext_trigger_callback: {e:?}");
    }
  }
  let timestamp_begin_end = (camera_data.timestamp_begin_end.load(Ordering::Relaxed) << 32) | timestamp as u64;
  camera_data.ext_trigger_counter.fetch_add(1, Ordering::Relaxed);
  camera_data.timestamp_begin_end.store(timestamp_begin_end, Ordering::Relaxed);
  camera_data_weak
}
