use std::boxed::Box;
use std::ffi::CString;
use std::path::PathBuf;
use std::process::abort;
use std::f32::consts::PI;
use std::fs::read_to_string;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use clap::Parser;
use numpy::PyArray;
use nalgebra as na;
use ndarray::prelude::*;
use tracing_panic::panic_hook;
use tracing::{info, warn, error};
use anyhow::{Error, Result, bail, ensure};
use pyo3::{prelude::*, types::PyTuple, ffi::c_str};
use tracing_subscriber::{fmt, EnvFilter, prelude::*};
use libredr_common::add_config;
#[cfg(feature = "loader_render")]
use event_ps::loader::Render;
#[cfg(feature = "loader_prophesee")]
use event_ps::loader::Prophesee;
use event_ps::{get_scan_pattern, default_config, ps_cl::PSCL, loader::{Loader, EventReader}};

#[cfg(feature = "display_cv")]
static CV_MUTEX: Mutex<()> = Mutex::new(());

#[derive(Parser)]
#[command(author = "Bohan Yu <ybh1998@protonmail.com>", about = "EventPS")]
struct Args {
  #[arg(num_args(1..), required(true), help = "Config files in `ini` format")]
  configs: Vec<PathBuf>,
}

fn normal_benchmark(normal: ArrayView3<f32>, normal_gt: ArrayView3<f32>) -> Result<()> {
  let shape = normal.shape();
  ensure!(shape == normal_gt.shape());
  ensure!(shape.len() == 3);
  ensure!(shape[0] == 3);
  let normal = normal.to_shape((3, shape[1] * shape[2]))?;
  let normal_gt = normal_gt.to_shape((3, shape[1] * shape[2]))?;
  let iter = normal.axis_iter(Axis(1)).zip(normal_gt.axis_iter(Axis(1)));
  let (n_pixels, ang_err_sum) = iter.fold((0, 0.), |(n_pixels, ang_err_sum), (normal, normal_gt)| {
    let normal = na::Vector3::new(normal[0], normal[1], normal[2]);
    let normal_gt = na::Vector3::new(normal_gt[0], normal_gt[1], normal_gt[2]);
    if normal_gt.norm() < 1e-3 {
      (n_pixels, ang_err_sum)
    } else {
      (n_pixels + 1, ang_err_sum + normal.dot(&normal_gt))
    }
  });
  let ang_err_mean = (ang_err_sum / (n_pixels as f32 + 1e-6)).acos() / PI * 180.;
  println!("BENCHMARK Event-LS-PS n_pixels {n_pixels} ang_err_mean {ang_err_mean}");
  Ok(())
}

fn py_add_data_eval<'py>(
    py: Python<'py>,
    py_module: &Py<PyModule>,
    args: impl IntoPyObject<'py, Target = PyTuple>,
    method: &str) -> Result<()> {
  let add_data_eval = py_module.getattr(py, "add_data_eval")?;
  add_data_eval.call1(py, args).map(|ret| {
    if let Ok((n_pixels, ang_err_mean)) = ret.extract::<(u32, f32)>(py) {
      println!("BENCHMARK {method} n_pixels {n_pixels} ang_err_mean {ang_err_mean}");
    }
  }).map_err(|e| {
    error!("py_add_data_eval: {method}: {e:?}");
    Error::from(e)
  })
}

fn run() -> Result<()> {
  let args = Args::parse();
  let mut config: HashMap<String, HashMap<String, String>> = default_config();
  for config_file in args.configs {
    println!("main: Adding config: `{}`", config_file.display());
    add_config(&mut config, config_file.as_ref())?;
  }
  println!("Configuration: {:#?}", config);
  let log_level = config["main"]["log_level"].to_owned().to_lowercase();
  let fmt_layer = fmt::layer()
    .with_target(false);
  let filter_layer = EnvFilter::try_new(log_level)?;
  tracing_subscriber::registry()
    .with(filter_layer)
    .with(fmt_layer)
    .init();
  let (loader, arr_ev_chunk_receiver, n_row, n_col) = match config["main"]["loader"].as_str() {
    "render" => {
      #[cfg(feature = "loader_render")]
      {
        let config_loader_render = &config["loader_render"];
        let (loader, arr_ev_chunk_receiver) = Render::new(config_loader_render,
                                                          #[cfg(feature = "display_cv")] &CV_MUTEX)?;
        let n_row = config_loader_render["height"].parse::<u16>()?;
        let n_col = config_loader_render["width"].parse::<u16>()?;
        let loader: Box<dyn Loader + Send> = Box::new(loader);
        (loader, arr_ev_chunk_receiver, n_row, n_col)
      }
      #[cfg(not(feature = "loader_render"))]
      {
        bail!("main: `loader_render` feature is not enabled in this compile");
      }
    },
    "event_reader" => {
      let config_loader_event_reader = &config["loader_event_reader"];
      let (loader, arr_ev_chunk_receiver) = EventReader::new(config_loader_event_reader,
                                                             #[cfg(feature = "display_cv")] &CV_MUTEX)?;
      let n_row = config_loader_event_reader["height"].parse::<u16>()?;
      let n_col = config_loader_event_reader["width"].parse::<u16>()?;
      let loader: Box<dyn Loader + Send> = Box::new(loader);
      (loader, arr_ev_chunk_receiver, n_row, n_col)
    },
    "prophesee" => {
      #[cfg(feature = "loader_prophesee")]
      {
        let config_loader_prophesee = &config["loader_prophesee"];
        let (loader, arr_ev_chunk_receiver) = Prophesee::new(config_loader_prophesee,
                                                             #[cfg(feature = "display_cv")] &CV_MUTEX)?;
        info!("Connected camera, waiting for external trigger!");
        let begin_row = config_loader_prophesee["begin_row"].parse::<u16>()?;
        let end_row = config_loader_prophesee["end_row"].parse::<u16>()?;
        let begin_col = config_loader_prophesee["begin_col"].parse::<u16>()?;
        let end_col = config_loader_prophesee["end_col"].parse::<u16>()?;
        let loader: Box<dyn Loader + Send> = Box::new(loader);
        (loader, arr_ev_chunk_receiver, end_row - begin_row, end_col - begin_col)
      }
      #[cfg(not(feature = "loader_prophesee"))]
      {
        bail!("main: `loader_prophesee` feature is not enabled in this compile");
      }
    },
    loader => {
      bail!("main: Unexpected `loader` type: {loader}");
    },
  };
  let loader = Arc::new(Mutex::new(Some(loader)));
  let loader_handler = loader.to_owned();
  let mut counter_handler = 0;
  ctrlc::set_handler(move || {
    if counter_handler == 0 {
      counter_handler = 1;
      info!("main: Stopping camera, Ctrl+C again to abort");
      let mut loader_guard = loader_handler.lock().expect("No task should panic");
      let _ = loader_guard.take();
    } else {
      error!("main: Ctrl+C received");
      abort();
    }
  })?;
  let load_normal = if config["main"]["loader"].as_str() != "event_reader" {
    ""
  } else {
    config["loader_render"]["load_normal"].as_str()
  };
  let normal_gt = if load_normal.is_empty() {
    None
  } else {
    Some(event_ps::load_normal(load_normal, n_row as usize, n_col as usize)?)
  };
  let normal_gt_python = Python::with_gil(|py| {
    Ok::<_, Error>(if load_normal.is_empty() {
      py.None()
    } else {
      event_ps::py_load_normal(py, load_normal, n_row as usize, n_col as usize)?.into_any().unbind()
    })
  })?;
  let config_ps = &config["ps"];
  let record_time = config_ps["record_time"].parse::<u32>()?;
  let record_n_bin = config_ps["record_n_bin"].parse::<u32>()?;
  let time_per_bin = record_time / record_n_bin;
  let ps_fcn_per_n_bin = config_ps["ps_fcn_per_n_bin"].parse::<u32>()?;
  let cnn_ps_per_n_bin = config_ps["cnn_ps_per_n_bin"].parse::<u32>()?;
  let mut ps_cl = PSCL::new(n_row, n_col, config_ps, arr_ev_chunk_receiver.to_owned())?;
  let code_path = config_ps["ps_fcn_python"].as_str();
  let ps_fcn_python: Option<Py<PyModule>> = (ps_fcn_per_n_bin > 0 && !code_path.is_empty()).then(|| {
    Python::with_gil(|py| {
      let code = CString::new(read_to_string(code_path)?)?;
      let code_path = CString::new(code_path)?;
      let module = PyModule::from_code(py, code.as_c_str(), code_path.as_c_str(), c_str!("__ev_ps_fcn_main__"))?;
      Ok::<_, Error>(module.into())
    })
  }).transpose()?;
  let code_path = config_ps["cnn_ps_python"].as_str();
  let cnn_ps_python: Option<Py<PyModule>> = (cnn_ps_per_n_bin > 0 && !code_path.is_empty()).then(|| {
    Python::with_gil(|py| {
      let code = CString::new(read_to_string(code_path)?)?;
      let code_path = CString::new(code_path)?;
      let module = PyModule::from_code(py, code.as_c_str(), code_path.as_c_str(), c_str!("__ev_cnn_ps_main__"))?;
      Ok::<_, Error>(module.into())
    })
  }).transpose()?;
  let mut last_cd_counter = 0;
  let mut last_ext_trigger_counter = 0;
  let interval = Duration::from_micros(1000000);
  let mut next_time = Instant::now() + interval;
  let mut fps_counter = 0;
  loop {
    match ps_cl.render_ls_ps(#[cfg(feature = "display_cv")] &CV_MUTEX) {
      Err(err) => {
        warn!("PSCL::render_ls_ps: {err:?}");
        return Ok(());
      },
      Ok(Some(render_buffer)) => {
        let normal = render_buffer.view().split_at(Axis(2), n_col as usize).1;
        let normal = normal.as_standard_layout().into_owned();
        if let Some(normal_gt) = &normal_gt {
          normal_benchmark(normal.view(), normal_gt.view())?;
        }
      },
      _ => (),
    }
    if ps_fcn_per_n_bin > 0 {
      match ps_cl.render_ps_fcn() {
        Ok((buffer, timestamp_max)) => {
          if let Some(ps_fcn_python) = &ps_fcn_python {
            let scan_pattern = get_scan_pattern(&config["loader_render"])?;
            let mut light_dir = Array2::default((buffer.shape()[0], 3));
            light_dir.outer_iter_mut().enumerate().for_each(|(i_bin, mut light_dir)| {
              let time = (timestamp_max - time_per_bin as usize * i_bin) as f32 * 1e-6;
              let light_dir_vector = scan_pattern(time);
              light_dir[0] = light_dir_vector[0];
              light_dir[1] = light_dir_vector[1];
              light_dir[2] = light_dir_vector[2];
            });
            Python::with_gil(|py| {
              let buffer_python = PyArray::from_owned_array(py, buffer);
              let light_dir_python = PyArray::from_owned_array(py, light_dir);
              let args = (buffer_python, light_dir_python, normal_gt_python.bind(py), &config_ps["mask"]);
              py_add_data_eval(py, ps_fcn_python, args, "Event-PS-FCN")
            })?;
          }
        },
        Err(err) => {
          warn!("PSCL::render_ps_fcn: {err:?}");
          return Ok(());
        }
      }
    }
    if cnn_ps_per_n_bin > 0 {
      match ps_cl.render_cnn_ps() {
        Ok(buffer) => {
          if let Some(cnn_ps_python) = &cnn_ps_python {
            Python::with_gil(|py| {
              let buffer_python = PyArray::from_owned_array(py, buffer);
              let args = (buffer_python, normal_gt_python.bind(py), &config_ps["mask"]);
              py_add_data_eval(py, cnn_ps_python, args, "Event-CNN-PS")
            })?;
          }
        },
        Err(err) => {
          warn!("PSCL::render_cnn_ps: {err:?}");
          return Ok(());
        }
      }
    }
    fps_counter += 1;
    if next_time <= Instant::now() {
      let event_info = {
        let loader_guard = loader.lock().expect("No task should panic");
        if let Some(loader) = &*loader_guard {
          let cd_counter = loader.get_cd_counter();
          let event_rate = (cd_counter - last_cd_counter) as f32 * 1e-6f32 / interval.as_secs_f32();
          last_cd_counter = cd_counter;
          let mut event_info = format!("Event rate: {:>6.2} Mev/s", event_rate);
          if let Some(ext_trigger_counter) = loader.get_ext_trigger_counter() {
            let rotate_speed = (ext_trigger_counter - last_ext_trigger_counter) as f32 * 60f32 / interval.as_secs_f32();
            last_ext_trigger_counter = ext_trigger_counter;
            event_info = format!("{event_info}, Rotate speed: {:>5.0} rpm", rotate_speed);
          }
          event_info
        } else {
          String::from("Camera stopped, waiting for channels to finish")
        }
      };
      info!("{event_info}, PS fps (V-Sync): {:>5.1}", fps_counter as f32 / interval.as_secs_f32());
      for ev_chunk_receiver in &arr_ev_chunk_receiver {
        if ev_chunk_receiver.is_full() {
          warn!("`ev_chunk` channel full!");
        }
      }
      next_time += interval;
      fps_counter = 0;
    }
  }
}

fn main() {
  std::panic::set_hook(Box::new(panic_hook));
  if let Err(e) = run() {
    panic!("main: {e:?}");
  }
}
