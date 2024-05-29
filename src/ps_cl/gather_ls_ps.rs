use std::mem::size_of;
use std::slice::from_raw_parts_mut;
use anyhow::Result;
use ndarray::ArrayViewMut3;
use ocl::{ProQue, Buffer, Kernel};
use crossbeam_channel::{Sender, Receiver};
use super::EV_BUFFER_N_JOBS;

pub struct GatherLSPS {
  arr_kernel: [Kernel; EV_BUFFER_N_JOBS],
  arr_request_sender: [Sender<Buffer<u8>>; EV_BUFFER_N_JOBS],
  response_receiver: Receiver<(u16, u32, Buffer<u8>)>,
}

impl GatherLSPS {
  pub fn new(
      n_row: u16,
      n_col: u16,
      proque: &ProQue,
      cl_render_buffer: &ocl::core::Mem,
      arr_request_sender: [Sender<Buffer<u8>>; EV_BUFFER_N_JOBS],
      response_receiver: Receiver<(u16, u32, Buffer<u8>)>) -> Result<Self> {
    let arr_kernel = (0..EV_BUFFER_N_JOBS).map(|i_job| {
      let kernel_ls_ps = proque.kernel_builder("kernel_ls_ps")
        .arg(i_job as u16)
        .arg(0u32)
        .arg(None::<&Buffer<u8>>)
        .arg(None::<&Buffer<u8>>)
        .global_work_size((n_row as usize, n_col as usize / EV_BUFFER_N_JOBS))
        .build()?;
      ocl::core::set_kernel_arg(kernel_ls_ps.as_core(), 3, ocl::core::ArgVal::mem(cl_render_buffer))?;
      Ok(kernel_ls_ps)
    }).collect::<Result<Vec<Kernel>>>()?.try_into().expect("EV_BUFFER_N_JOBS");
    Ok(GatherLSPS {
      arr_kernel,
      arr_request_sender,
      response_receiver,
    })
  }

  pub fn gather(&mut self,
      proque: &ProQue,
      cl_render_buffer: &ocl::core::Mem,
      render_buffer: Option<ArrayViewMut3<f32>>) -> Result<()> {
    let arr_pixel_buffer: [Option<(u16, Buffer<u8>)>; EV_BUFFER_N_JOBS] =
      (0..EV_BUFFER_N_JOBS).map(|_| {
        let Ok((i_job, timestamp_max, pixel_buffer)) = self.response_receiver.try_recv() else {
          return Ok(None);
        };
        self.arr_kernel[i_job as usize].set_arg(1, timestamp_max)?;
        self.arr_kernel[i_job as usize].set_arg(2, &pixel_buffer)?;
        unsafe { self.arr_kernel[i_job as usize].enq()?; }
        Ok(Some((i_job, pixel_buffer)))
      }).collect::<Result<Vec<Option<(u16, Buffer<u8>)>>>>()?.try_into().expect("EV_BUFFER_N_JOBS");
    if let Some(mut render_buffer) = render_buffer {
      unsafe {
        let render_buffer_slice = from_raw_parts_mut(render_buffer.as_mut_ptr() as *mut u8,
                                                     render_buffer.len() * size_of::<f32>());
        ocl::core::enqueue_read_buffer(proque.as_core(),
                                       cl_render_buffer,
                                       true,
                                       0,
                                       render_buffer_slice,
                                       None::<ocl::core::Event>,
                                       None::<&mut ocl::core::Event>)?;
      };
    }
    arr_pixel_buffer.into_iter().map(|pixel_buffer| {
      let Some((i_job, pixel_buffer)) = pixel_buffer else {
        return Ok(());
      };
      self.arr_request_sender[i_job as usize].send(pixel_buffer)?;
      Ok(())
    }).collect::<Result<Vec<()>>>().map(|_| ())
  }
}
