use std::boxed::Box;
#[cfg(feature = "display_cv")]
use std::sync::Mutex;
use std::time::Duration;
use std::collections::HashMap;
use std::mem::{size_of, take};
use std::{thread, thread::JoinHandle};
#[cfg(feature = "display_gl")]
use glfw::Context;
#[cfg(feature = "display_cv")]
use ndarray::Axis;
use tracing::{info, error};
use crossbeam_channel::Receiver;
use ndarray::{Array3, Array4, Array5};
use anyhow::{Result, anyhow, bail, ensure};
#[cfg(feature = "display_gl")]
use ocl::builders::{ClWaitListPtrEnum, ClNullEventPtrEnum};
use ocl::{Platform, Device, Program, ProQue, builders::{ContextBuilder, ProgramBuilder}};
use super::loader::{EventChunk, EV_BUFFER_N_JOBS, EV_BUFFER_N_BUCKET, EV_BUFFER_BUCKET_SIZE};
use self::gather_ls_ps::GatherLSPS;
use self::gather_ps_fcn_cnn_ps::GatherPSFCNCNNPS;
use self::process_chunk::spawn_process_chunk;
use self::jit_kernel_sources::jit_kernel_sources;

mod gather_ls_ps;
mod gather_ps_fcn_cnn_ps;
mod process_chunk;
mod jit_kernel_sources;

const OBS_MAP_RESOLUTION: usize = 32;

fn get_platform_device(config_ps: &HashMap<String, String>) -> Result<(Platform, Device)>{
  let platform_id: usize = config_ps["opencl_platform_id"].parse()?;
  let device_id: usize = config_ps["opencl_device_id"].parse()?;
  let platform_list = Platform::list();
  let Some(platform) = platform_list.get(platform_id) else {
    bail!("Error: RenderCL::new: platform_id {platform_id} exceeds maximum {}", platform_list.len());
  };
  let device_list = Device::list(platform, None)?;
  let Some(device) = device_list.get(device_id) else {
    bail!("Error: RenderCL::new: device_id {device_id} exceeds maximum {}", device_list.len());
  };
  if cfg!(target_endian = "little") {
    let ocl::enums::DeviceInfoResult::EndianLittle(is_endian_little) =
      device.info(ocl::enums::DeviceInfo::EndianLittle)? else {
      unreachable!("required for endian");
    };
    ensure!(is_endian_little,
      "Error: RenderCL::get_platform_device: Host is little endian but \
      platform_id {platform_id} device_id {device_id} is big endian");
  } else {
    let ocl::enums::DeviceInfoResult::EndianLittle(is_endian_little) =
      device.info(ocl::enums::DeviceInfo::EndianLittle)? else {
      unreachable!("required for endian");
    };
    ensure!(!is_endian_little,
      "Error: RenderCL::get_platform_device: Host is big endian but \
      platform_id {platform_id} device_id {device_id} is little endian");
  }
  Ok((platform.to_owned(), device.to_owned()))
}

pub struct PSCL {
  _n_row: u16,
  _n_col: u16,
  show_ls_ps: String,
  // All OpenCL component must be dropped before OpenGL
  _proque: ProQue,
  cl_render_buffer: ocl::core::Mem,
  gather_ls_ps: Option<GatherLSPS>,
  gather_ps_fcn: Option<GatherPSFCNCNNPS>,
  gather_cnn_ps: Option<GatherPSFCNCNNPS>,
  #[cfg(feature = "display_gl")]
  gl_window: Option<glfw::PWindow>,
  #[cfg(feature = "display_gl")]
  gl_fbo: Option<gl::types::GLuint>,
  // Must be `Vec` to be `take` during drop
  vec_task_process_chunk: Vec<JoinHandle<()>>,
}

impl PSCL {
  #[cfg(feature = "display_gl")]
  fn gl_create_window(glfw: &mut glfw::Glfw, n_row: u16, n_col: u16, title: &str) -> Result<glfw::PWindow> {
    glfw.window_hint(glfw::WindowHint::Resizable(false));
    let window = glfw.create_window(n_col as u32, n_row as u32, title, glfw::WindowMode::Windowed);
    let Some((window, _)) = window else {
      bail!("glfw failed to create window {title}");
    };
    Ok(window)
  }

  #[cfg(feature = "display_gl")]
  fn gl_init(
      n_row: u16,
      n_col: u16,
      mut context_builder: ContextBuilder,
      program_builder: ProgramBuilder,
      ) -> Result<(ProQue, Option<glfw::PWindow>, Option<gl::types::GLuint>, ocl::core::Mem)> {
    use glfw::fail_on_errors;
    let mut glfw = glfw::init(fail_on_errors!()).unwrap();
    let mut gl_window = Self::gl_create_window(&mut glfw, n_row, n_col * 2, "ev_ps")?;
    gl_window.make_current();
    // V-Sync
    glfw.set_swap_interval(glfw::SwapInterval::Sync(1));
    // Unlimited (for benchmarking)
    // glfw.set_swap_interval(glfw::SwapInterval::None);
    gl::load_with(|s| gl_window.get_proc_address(s));
    let gl_context = gl_window.get_glx_context();
    let glx_display = glfw.get_x11_display();
    info!("Creating GL context {gl_context:?} {glx_display:?}");
    // TODO: find the same CL device as GL
    let context = context_builder
      .property(ocl::enums::ContextPropertyValue::GlContextKhr(gl_context))
      .property(ocl::enums::ContextPropertyValue::GlxDisplayKhr(glx_display))
      .build()?;
    let mut gl_fbo: gl::types::GLuint = 0;
    let mut gl_render_buffer: gl::types::GLuint = 0;
    unsafe {
      gl::GenFramebuffers(1, &mut gl_fbo as *mut gl::types::GLuint);
      gl::BindFramebuffer(gl::READ_FRAMEBUFFER, gl_fbo);
      gl::GenRenderbuffers(1, &mut gl_render_buffer as *mut gl::types::GLuint);
      gl::BindRenderbuffer(gl::RENDERBUFFER, gl_render_buffer);
      gl::RenderbufferStorage(gl::RENDERBUFFER, gl::RGBA8, (n_col * 2) as i32, n_row as i32);
      gl::FramebufferRenderbuffer(gl::READ_FRAMEBUFFER, gl::COLOR_ATTACHMENT0, gl::RENDERBUFFER, gl_render_buffer);
    }
    let cl_render_buffer = unsafe {
      ocl::core::create_from_gl_renderbuffer(&context, gl_render_buffer, ocl::core::MEM_WRITE_ONLY)?
    };
    let proque = ProQue::builder().context(context).prog_bldr(program_builder).build()?;
    Ok((proque, Some(gl_window), Some(gl_fbo), cl_render_buffer))
  }

  fn cv_none_init(
      n_row: u16,
      n_col: u16,
      context_builder: ContextBuilder,
      program_builder: ProgramBuilder) -> Result<(ProQue, ocl::core::Mem)> {
    let context = context_builder.build()?;
    let render_buffer_size = 3 * n_row as usize * 2 * n_col as usize * size_of::<f32>();
    let cl_render_buffer = unsafe {
      ocl::core::create_buffer(&context, ocl::core::MEM_WRITE_ONLY, render_buffer_size, Option::<&[u8]>::None)?
    };
    let proque = ProQue::builder().context(context).prog_bldr(program_builder).build()?;
    Ok((proque, cl_render_buffer))
  }

  pub fn new(
      n_row: u16,
      n_col: u16,
      config_ps: &HashMap<String, String>,
      arr_ev_chunk_receiver: [Receiver<Box<EventChunk>>; EV_BUFFER_N_JOBS]) -> Result<PSCL> {
    let show_ls_ps = config_ps["show_ls_ps"].to_owned();
    let source = jit_kernel_sources(n_row, n_col, config_ps)?;
    let mut program_builder = Program::builder();
    program_builder.src(source)
      .cmplr_opt("-cl-mad-enable")
      .cmplr_opt("-cl-fast-relaxed-math")
      .cmplr_opt("-cl-no-signed-zeros")
      .cmplr_opt("-Werror")
      .cmplr_opt("-cl-std=CL1.2");
    let (platform, device) = get_platform_device(config_ps)?;
    info!("PSCL::new: Platform: {}, Device: {}.", platform.name()?, device.name()?);
    let mut context_builder = ocl::Context::builder();
    context_builder.platform(platform).devices(device);
    #[cfg(feature = "display_gl")]
    let (proque, gl_window, gl_fbo, cl_render_buffer) = match show_ls_ps.as_str() {
      "gl" => Self::gl_init(n_row, n_col, context_builder, program_builder)?,
      "cv" | "none" => {
        let (proque, cl_render_buffer) = Self::cv_none_init(n_row, n_col, context_builder, program_builder)?;
        (proque, None, None, cl_render_buffer)
      },
      show_ls_ps => bail!("PSCL::new: Unexpected `show_ls_ps` type: {show_ls_ps}"),
    };
    #[cfg(not(feature = "display_gl"))]
    let (proque, cl_render_buffer) = match show_ls_ps.as_str() {
      "gl" => bail!("PSCL::new: `display_gl` feature is not enabled in this compile"),
      "cv" | "none" => Self::cv_none_init(n_row, n_col, context_builder, program_builder)?,
      show_ls_ps => bail!("PSCL::new: Unexpected `show_ls_ps` type: {show_ls_ps}"),
    };
    let (vec_task_process_chunk, gather_ls_ps, gather_ps_fcn, gather_cnn_ps) =
      spawn_process_chunk(n_row, n_col, config_ps, arr_ev_chunk_receiver, &proque, &cl_render_buffer)?;
    Ok(PSCL {
      _n_row: n_row,
      _n_col: n_col,
      show_ls_ps,
      _proque: proque,
      cl_render_buffer,
      gather_ls_ps,
      gather_ps_fcn,
      gather_cnn_ps,
      #[cfg(feature = "display_gl")]
      gl_window,
      #[cfg(feature = "display_gl")]
      gl_fbo,
      vec_task_process_chunk,
    })
  }

  fn finish(&mut self) {
    info!("PSCL::finish: Waiting for all `task_process_chunk` to finish");
    take(&mut self.vec_task_process_chunk).into_iter().for_each(|task_process_chunk| {
      if let Err(e) = task_process_chunk.join() {
        error!("PSCL::finish: {e:?}");
      }
    });
    info!("PSCL::finish: All `task_process_chunk` finished");
  }

  #[cfg(feature = "display_gl")]
  fn gl_acquire(&mut self) -> Result<()> {
    let Some(gl_window) = &mut self.gl_window else {
      bail!("PSCL::gl_acquire: `gl_window` is None");
    };
    gl_window.make_current();
    gl::load_with(|s| gl_window.get_proc_address(s));
    unsafe {
      gl::Flush();
      gl::Finish();
    }
    ocl::core::enqueue_acquire_gl_objects::<ClNullEventPtrEnum, ClWaitListPtrEnum>(
      self._proque.queue().as_core(),
      std::slice::from_ref(&self.cl_render_buffer),
      None,
      None)?;
    Ok(())
  }

  #[cfg(feature = "display_gl")]
  fn gl_release(&mut self) -> Result<()> {
    let (Some(gl_window), Some(gl_fbo)) = (&mut self.gl_window, self.gl_fbo) else {
      bail!("PSCL::gl_release: `gl_window` or `gl_fbo` is None");
    };
    unsafe {
      gl::BindFramebuffer(gl::READ_FRAMEBUFFER, gl_fbo);
      gl::ReadBuffer(gl::COLOR_ATTACHMENT0);
      gl::BindFramebuffer(gl::DRAW_FRAMEBUFFER, 0);
      gl::DrawBuffer(gl::BACK_LEFT);
      gl::BlitFramebuffer(0, 0, (self._n_col * 2) as i32, self._n_row as i32,
                          0, 0, (self._n_col * 2) as i32, self._n_row as i32,
                          gl::COLOR_BUFFER_BIT, gl::NEAREST);
    }
    gl_window.swap_buffers();
    Ok(())
  }

  pub fn render_ls_ps(
      &mut self,
      #[cfg(feature = "display_cv")] cv_mutex: &'static Mutex<()>) -> Result<Option<Array3<f32>>> {
    #[cfg(feature = "display_gl")]
    if matches!(self.show_ls_ps.as_str(), "gl") {
      self.gl_acquire()?;
    }
    let (render_buffer, ret) = if let Some(gather_ls_ps) = &mut self.gather_ls_ps {
      thread::sleep(Duration::from_millis(10));
      let mut render_buffer = if matches!(self.show_ls_ps.as_str(), "cv" | "none") {
        Some(Array3::zeros((3, self._n_row as usize, 2 * self._n_col as usize)))
      } else {
        None
      };
      let ret = gather_ls_ps.gather(&self._proque,
                                    &self.cl_render_buffer,
                                    render_buffer.as_mut().map(|render_buffer| render_buffer.view_mut()));
      (render_buffer, ret)
    } else {
      (None, Ok(()))
    };
    match self.show_ls_ps.as_str() {
      "gl" => {
        #[cfg(feature = "display_gl")]
        {
          ocl::core::enqueue_release_gl_objects::<ClNullEventPtrEnum, ClWaitListPtrEnum>(
            self._proque.queue().as_core(),
            std::slice::from_ref(&self.cl_render_buffer),
            None,
            None)?;
          self._proque.queue().flush()?;
          self._proque.queue().finish()?;
          self.gl_release()?;
        }
        #[cfg(not(feature = "display_gl"))]
        unreachable!()
      },
      "cv" => {
        #[cfg(feature = "display_cv")]
        {
          let render_buffer = render_buffer.as_ref().expect("show_ls_ps == cv");
          let (event, normal) = render_buffer.view().split_at(Axis(2), self._n_col as usize);
          crate::cv_show_image("Event-LS-PS event", event, cv_mutex)?;
          crate::cv_show_normal("Event-LS-PS normal", normal, cv_mutex)?;
        }
        #[cfg(not(feature = "display_cv"))]
        bail!("PSCL::render_ls_ps: `display_cv` feature is not enabled in this compile")
      },
      _ => (),
    }
    // Process error after `release_gl_objects`
    ret.map_err(|_| { self.finish(); anyhow!("One `arr_gather_ls_ps_request_sender` closed") })?;
    for task_process_chunk in &self.vec_task_process_chunk {
      if task_process_chunk.is_finished() {
        self.finish();
        bail!("One `task_process_chunk` finished");
      }
    }
    Ok(render_buffer)
  }

  pub fn render_ps_fcn(&mut self) -> Result<(Array4<f32>, usize)> {
    let Some(gather_ps_fcn) = &mut self.gather_ps_fcn else {
      bail!("render_ps_fcn: `ps_fcn` not enabled");
    };
    let (buffer, timestamp_max) = gather_ps_fcn.gather()?;
    Ok((buffer.into_dimensionality()?, timestamp_max))
  }

  pub fn render_cnn_ps(&mut self) -> Result<Array5<f32>> {
    let Some(gather_cnn_ps) = &mut self.gather_cnn_ps else {
      bail!("render_cnn_ps: `cnn_ps` not enabled");
    };
    Ok(gather_cnn_ps.gather()?.0.into_dimensionality()?)
  }
}
