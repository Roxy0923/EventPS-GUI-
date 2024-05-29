use std::mem::size_of;
use std::collections::HashMap;
use std::slice::from_raw_parts_mut;
use itertools::izip;
use ndarray::prelude::*;
use anyhow::{Result, ensure};
use crossbeam_channel::{Sender, Receiver};
use ocl::{ProQue, Buffer, MemFlags, Kernel, EventList};
use super::{EV_BUFFER_N_JOBS, OBS_MAP_RESOLUTION};

pub struct GatherPSFCNCNNPS {
  buffer: Buffer<u8>,
  arr_kernel: [Kernel; EV_BUFFER_N_JOBS],
  arr_request_sender: [Sender<Buffer<u8>>; EV_BUFFER_N_JOBS],
  arr_response_receiver: [Receiver<Buffer<u8>>; EV_BUFFER_N_JOBS],
  timestamp_max: usize,
  per_timestamp: u32,
  gather_shape: Vec<usize>,
}

impl GatherPSFCNCNNPS {
  pub fn new(
      method: &str,
      n_row: u16,
      n_col: u16,
      config_ps: &HashMap<String, String>,
      proque: &ProQue,
      arr_request_sender: [Sender<Buffer<u8>>; EV_BUFFER_N_JOBS],
      arr_response_receiver: [Receiver<Buffer<u8>>; EV_BUFFER_N_JOBS]) -> Result<Self> {
    let record_time = config_ps["record_time"].parse::<u32>()?;
    let record_n_bin = config_ps["record_n_bin"].parse::<u32>()?;
    let time_per_bin = record_time / record_n_bin;
    let per_n_bin = config_ps[&format!("{method}_per_n_bin")].parse::<u32>()?;
    let per_timestamp = time_per_bin * per_n_bin;
    let record_n_bin = config_ps["record_n_bin"].parse::<usize>()?;
    let gather_shape = match method {
      "ps_fcn" => vec![record_n_bin - 2, 3, n_row as usize, n_col as usize],
      "cnn_ps" => vec![3, OBS_MAP_RESOLUTION, OBS_MAP_RESOLUTION, n_row as usize, n_col as usize],
      _ => unreachable!(),
    };
    let buffer = Buffer::builder()
      .queue(proque.queue().to_owned())
      .flags(MemFlags::WRITE_ONLY)
      .len(gather_shape.iter().product::<usize>() * size_of::<f32>())
      .build()?;
    let arr_kernel = (0..EV_BUFFER_N_JOBS).map(|i_job| {
      let kernel = proque.kernel_builder(format!("kernel_{method}"))
        .arg(i_job as u16)
        .arg(0u32)
        .arg(None::<&Buffer<u8>>)
        .arg(&buffer)
        .global_work_size((n_row as usize, n_col as usize / EV_BUFFER_N_JOBS))
        .build()?;
      Ok(kernel)
    }).collect::<Result<Vec<Kernel>>>()?.try_into().expect("EV_BUFFER_N_JOBS");
    Ok(GatherPSFCNCNNPS {
      buffer,
      arr_kernel,
      arr_request_sender,
      arr_response_receiver,
      timestamp_max: (record_time - 1 - per_timestamp) as usize,
      per_timestamp,
      gather_shape,
    })
  }

  pub fn gather(&mut self) -> Result<(ArrayD<f32>, usize)> {
    ensure!(self.per_timestamp > 0, "GatherPSFCNCNNPS::gather: per_n_bin = 0");
    self.timestamp_max += self.per_timestamp as usize;
    let mut event_execution_list = EventList::with_capacity(EV_BUFFER_N_JOBS);
    let arr_pixel_buffer: [Buffer<u8>; EV_BUFFER_N_JOBS] =
      izip!(&self.arr_response_receiver, &mut self.arr_kernel).map(|(response_receiver, kernel)| {
        let pixel_buffer = response_receiver.recv()?;
        kernel.set_arg(1, self.timestamp_max as u32)?;
        kernel.set_arg(2, &pixel_buffer)?;
        unsafe { kernel.cmd().enew(&mut event_execution_list).enq()?; }
        Ok(pixel_buffer)
      }).collect::<Result<Vec<_>>>()?.try_into().expect("EV_BUFFER_N_JOBS");
    let mut gather_buffer = ArrayD::default(self.gather_shape.as_slice());
    // let mut gather_buffer = Array4::default([1, 2, 3, 4]);
    let gather_buffer_slice = unsafe {
      from_raw_parts_mut(gather_buffer.as_mut_ptr() as *mut u8, gather_buffer.len() * size_of::<f32>())
    };
    self.buffer.read(gather_buffer_slice).ewait(&event_execution_list).enq()?;
    izip!(&self.arr_request_sender, arr_pixel_buffer).map(|(request_sender, pixel_buffer)| {
      request_sender.send(pixel_buffer)?;
      Ok(())
    }).collect::<Result<Vec<_>>>()?;
    Ok((gather_buffer, self.timestamp_max))
  }
}
