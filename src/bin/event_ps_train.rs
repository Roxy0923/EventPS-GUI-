use std::boxed::Box;
#[cfg(feature = "display_cv")]
use std::sync::Mutex;
use std::ffi::CString;
use std::path::PathBuf;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::fs::{read_to_string, read_dir};
use fastrand::Rng;
use numpy::PyArray;
use ndarray::prelude::*;
use clap::{Args, Parser};
use anyhow::{Error, Result};
use tracing_panic::panic_hook;
use tracing::{info, warn, error};
use pyo3::{prelude::*, ffi::c_str};
use tracing_subscriber::{fmt, EnvFilter, prelude::*};
use libredr_common::add_config;
use event_ps::{get_scan_pattern, default_config, py_load_normal, ps_cl::PSCL, loader::{Loader, EventReader}};

#[cfg(feature = "display_cv")]
static CV_MUTEX: Mutex<()> = Mutex::new(());

#[derive(Parser)]
#[command(author = "Bohan Yu <ybh1998@protonmail.com>", about = "EventPS")]
struct Cli {
  #[command(flatten)]
  group: Group,
}

#[derive(Args)]
#[group(required = true, multiple = true)]
struct Group {
  #[arg(long, help = "ps_fcn python training code path")]
  ps_fcn_train: Option<PathBuf>,
  #[arg(long, help = "cnn_ps python training code path")]
  cnn_ps_train: Option<PathBuf>,
}

fn load_data(
    config_file: &PathBuf,
    ps_fcn_python: &Option<Py<PyModule>>,
    cnn_ps_python: &Option<Py<PyModule>>,
    rng: &mut Rng) -> Result<()> {
  info!("load_data: Loading {config_file:?}");
  let mut config: HashMap<String, HashMap<String, String>> = default_config();
  add_config(&mut config, config_file)?;
  println!("Configuration: {:#?}", config);
  let config_loader_render = &config["loader_render"];
  let config_loader_event_reader = HashMap::from([
    (String::from("width"), config_loader_render["width"].to_owned()),
    (String::from("height"), config_loader_render["height"].to_owned()),
    (String::from("load_event"), config["loader_event_reader"]["load_event"].to_owned()),
    (String::from("playback_speed"), String::from("0.")),
    (String::from("flush_interval"), String::from("2048")),
  ]);
  let (loader, arr_ev_chunk_receiver) = EventReader::new(&config_loader_event_reader,
                                                         #[cfg(feature = "display_cv")] &CV_MUTEX)?;
  let n_row = config_loader_event_reader["height"].parse::<u16>()?;
  let n_col = config_loader_event_reader["width"].parse::<u16>()?;
  let loader: Box<dyn Loader> = Box::new(loader);
  let mut config_ps = config["ps"].to_owned();
  let ps_fcn_per_n_bin = if ps_fcn_python.is_some() { 1 } else { 0 };
  config_ps.insert(String::from("ps_fcn_per_n_bin"), ps_fcn_per_n_bin.to_string());
  let cnn_ps_per_n_bin = if cnn_ps_python.is_some() { 4 } else { 0 };
  config_ps.insert(String::from("cnn_ps_per_n_bin"), cnn_ps_per_n_bin.to_string());
  let record_time = config_ps["record_time"].parse::<u32>()? * rng.u32(1..3);
  config_ps.insert(String::from("record_time"), record_time.to_string());
  let cnn_ps_half_life = (config_ps["cnn_ps_half_life"].parse::<f32>()? * (rng.f32() + 1.)) as u32;
  config_ps.insert(String::from("cnn_ps_half_life"), cnn_ps_half_life.to_string());
  let record_n_bin = config_ps["record_n_bin"].parse::<u32>()?;
  let time_per_bin = record_time / record_n_bin;
  let mut ps_cl = PSCL::new(n_row, n_col, &config_ps, arr_ev_chunk_receiver)?;
  let normal_gt_python = Python::with_gil(|py| {
    let normal_gt_python = py_load_normal(py,
                                          config_loader_render["save_normal"].as_str(),
                                          n_row as usize,
                                          n_col as usize)?;
    Ok::<_, Error>(normal_gt_python.unbind())
  })?;
  let mut last_cd_counter = 0;
  let interval = Duration::from_micros(1000000);
  let mut next_time = Instant::now() + interval;
  let mut fps_counter = 0;
  loop {
    if let Err(err) = ps_cl.render_ls_ps(#[cfg(feature = "display_cv")] &CV_MUTEX) {
      warn!("PSCL::render_ls_ps: {err:?}");
      return Ok(());
    }
    if ps_fcn_per_n_bin > 0 {
      match ps_cl.render_ps_fcn() {
        Ok((buffer, timestamp_max)) => {
          if let Some(ps_fcn_python) = ps_fcn_python {
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
              let add_data_train = ps_fcn_python.getattr(py, "add_data_train")?;
              let buffer_python = PyArray::from_owned_array(py, buffer);
              let light_dir_python = PyArray::from_owned_array(py, light_dir);
              let args = (buffer_python, light_dir_python, normal_gt_python.bind(py));
              if let Err(e) = add_data_train.call1(py, args) {
                error!("load_data: ps_fcn_python: {e:?}");
                return Err(e);
              }
              Ok(())
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
          if let Some(cnn_ps_python) = cnn_ps_python {
            Python::with_gil(|py| {
              let add_data_train = cnn_ps_python.getattr(py, "add_data_train")?;
              let buffer_python = PyArray::from_owned_array(py, buffer);
              let args = (buffer_python, normal_gt_python.bind(py));
              if let Err(e) = add_data_train.call1(py, args) {
                error!("load_data: cnn_ps_python: {e:?}");
                return Err(e);
              }
              Ok(())
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
      let cd_counter = loader.get_cd_counter();
      info!("Event rate: {:>8.3} Mev/s, PS FPS: {:>8.3}",
        (cd_counter - last_cd_counter) as f32 * 1e-6f32 / interval.as_secs_f32(),
        fps_counter as f32 / interval.as_secs_f32());
      last_cd_counter = cd_counter;
      next_time += interval;
      fps_counter = 0;
    }
  }
}

fn run() -> Result<()> {
  let cli = Cli::parse();
  let fmt_layer = fmt::layer()
    .with_target(false);
  let filter_layer = EnvFilter::try_new("info")?;
  tracing_subscriber::registry()
    .with(filter_layer)
    .with(fmt_layer)
    .init();
  let ps_fcn_python: Option<Py<PyModule>> = cli.group.ps_fcn_train.map(|ps_fcn_train| {
    Python::with_gil(|py| {
      let code = CString::new(read_to_string(&ps_fcn_train)?)?;
      let ps_fcn_train = CString::new(ps_fcn_train.to_str().unwrap())?;
      let module = PyModule::from_code(py, code.as_c_str(), ps_fcn_train.as_c_str(), c_str!("__ev_ps_fcn_main__"))?;
      Ok::<_, Error>(module.into())
    })
  }).transpose()?;
  let cnn_ps_python: Option<Py<PyModule>> = cli.group.cnn_ps_train.map(|cnn_ps_train| {
    Python::with_gil(|py| {
      let code = CString::new(read_to_string(&cnn_ps_train)?)?;
      let cnn_ps_train = CString::new(cnn_ps_train.to_str().unwrap())?;
      let module = PyModule::from_code(py, code.as_c_str(), cnn_ps_train.as_c_str(), c_str!("__ev_cnn_ps_main__"))?;
      Ok::<_, Error>(module.into())
    })
  }).transpose()?;
  let mut rng = Rng::with_seed(2);
  loop {
    let mut dir_training = Vec::new();
    dir_training.extend(read_dir("data/blobs_training/")?.into_iter());
    dir_training.extend(read_dir("data/sculptures_training/")?.into_iter());
    rng.shuffle(&mut dir_training);
    for dir_training in dir_training {
      let mut path = dir_training?.path();
      path.push("render.ini");
      load_data(&path, &ps_fcn_python, &cnn_ps_python, &mut rng)?;
    }
  }
}

fn main() {
  std::panic::set_hook(Box::new(panic_hook));
  if let Err(e) = run() {
    panic!("main: {e:?}");
  }
}
