pub mod ps_cl;
pub mod loader;

use std::io::Read;
use std::boxed::Box;
#[cfg(feature = "display_cv")]
use std::sync::Mutex;
use std::mem::size_of;
use std::f32::consts::PI;
use std::collections::HashMap;
use std::slice::from_raw_parts;
use nalgebra as na;
use pyo3::prelude::*;
use ndarray::prelude::*;
#[cfg(feature = "display_cv")]
use cv_convert::TryFromCv;
use anyhow::{Result, ensure};
use numpy::{PyArray, PyArray3};
#[cfg(feature = "display_cv")]
use opencv::{prelude::*, highgui::{imshow, poll_key}};

pub fn map_image(image: ArrayView3<f32>) -> Result<Array3<f32>> {
  ensure!(image.shape().len() == 3);
  ensure!(image.shape()[0] == 3);
  Ok(image.map(|x| x.powf(1. / 2.2).clamp(0., 1.)))
}

#[cfg(feature = "display_cv")]
pub fn cv_show_image(title: &str, image: ArrayView3<f32>, cv_mutex: &'static Mutex<()>) -> Result<()> {
  let image = Mat::try_from_cv(map_image(image)?.permuted_axes([1, 2, 0]).map(|x| (x * 255.) as u8))?;
  let guard = cv_mutex.lock().expect("No task should panic");
  imshow(title, &image)?;
  poll_key()?;
  drop(guard);
  Ok(())
}

pub fn map_normal(normal: ArrayView3<f32>) -> Result<Array3<f32>> {
  ensure!(normal.shape().len() == 3);
  ensure!(normal.shape()[0] == 3);
  let mut normal = normal.to_owned();
  normal.axis_iter_mut(Axis(2)).for_each(|mut normal| {
    normal.axis_iter_mut(Axis(1)).for_each(|mut normal| {
      normal.assign(&arr1(&[normal[2], -normal[0], normal[1]]));
    });
  });
  Ok(normal.map(|x| (0.5 + 0.5 * x).clamp(0., 1.)))
}

#[cfg(feature = "display_cv")]
pub fn cv_show_normal(title: &str, normal: ArrayView3<f32>, cv_mutex: &'static Mutex<()>) -> Result<()> {
  let normal = Mat::try_from_cv(map_normal(normal)?.permuted_axes([1, 2, 0]).map(|x| (x * 255.) as u8))?;
  let guard = cv_mutex.lock().expect("No task should panic");
  imshow(title, &normal)?;
  poll_key()?;
  drop(guard);
  Ok(())
}

pub fn load_normal(path: &str, n_row: usize, n_col: usize) -> Result<Array3::<f32>> {
  let mut buf = Vec::new();
  let buf_len = 3 * n_row * n_col;
  let mut reader = xz2::read::XzDecoder::new(std::fs::File::open(path)?);
  ensure!(reader.read_to_end(&mut buf)? == buf_len * size_of::<f32>());
  let buf_slice = unsafe { from_raw_parts(buf.as_ptr() as *const f32, buf_len) };
  Ok(ArrayView3::from_shape((3, n_row, n_col), buf_slice)?.to_owned())
}

pub fn py_load_normal(py: Python, path: &str, n_row: usize, n_col: usize) -> Result<Py<PyArray3::<f32>>> {
  Ok(PyArray::from_owned_array(py, load_normal(path, n_row, n_col)?).into())
}

pub fn get_scan_pattern(config: &HashMap<String, String>) -> Result<Box<dyn Fn(f32) -> na::Vector3<f32>>> {
  let duration = config["duration"].parse::<f32>()?;
  let n_rounds = config["n_rounds"].parse::<f32>()?;
  Ok(match config["scan_pattern"].as_str() {
    "circle" => {
      let latitude = config["circle_latitude"].parse::<f32>()? / 180. * PI;
      Box::new(move |time| {
        let longitude = time / duration * n_rounds * 2. * PI;
        na::Vector3::new(latitude.cos() * longitude.sin(),
                         latitude.cos() * longitude.cos(),
                         latitude.sin())
      })
    },
    "hypotrochoid" => {
      let r_big = config["hypotrochoid_r_big"].parse::<f32>()?;
      let start_big = config["hypotrochoid_start_big"].parse::<f32>()?;
      ensure!(r_big >= 0.);
      let r_small = config["hypotrochoid_r_small"].parse::<f32>()?;
      let start_small = config["hypotrochoid_start_small"].parse::<f32>()?;
      ensure!(r_small >= 0.);
      ensure!(r_big + r_small <= 1.);
      Box::new(move |time| {
        let longitude = time / duration * n_rounds * 2. * PI;
        let ratio: f32 = -1. / 1.5;
        let x = r_big * (ratio * longitude + start_big).sin() + r_small * (longitude + start_small).sin();
        let y = r_big * (ratio * longitude + start_big).cos() + r_small * (longitude + start_small).cos();
        let z = (1. - x.powi(2) - y.powi(2)).clamp(0., 1.).sqrt();
        na::Vector3::new(x, y, z)
      })
    },
    "diligent" => {
      Box::new(move |time| {
        let position = (time / duration * n_rounds) % 1.0;
        //  0.5740 -0.3580 0.7364 =>  0.7794 -0.4861 1.
        //  0.5465  0.3790 0.7468 =>  0.7317  0.5074 1.
        // -0.5495  0.3871 0.7404 => -0.7421  0.5228 1.
        // -0.5888 -0.3482 0.7294 => -0.8072 -0.4773 1.
        match position {
          position if position < ( 7. / 36.) => {
            let ratio = (position -  0. / 36.) / ( 7. / 36.);
            (1. - ratio) * na::Vector3::new( 0.7794, -0.4861, 1.) + ratio * na::Vector3::new( 0.7317,  0.5074, 1.)
          },
          position if position < (18. / 36.) => {
            let ratio = (position -  7. / 36.) / (11. / 36.);
            (1. - ratio) * na::Vector3::new( 0.7317,  0.5074, 1.) + ratio * na::Vector3::new(-0.7421,  0.5228, 1.)
          },
          position if position < (25. / 36.) => {
            let ratio = (position - 18. / 36.) / ( 7. / 36.);
            (1. - ratio) * na::Vector3::new(-0.7421,  0.5228, 1.) + ratio * na::Vector3::new(-0.8072, -0.4773, 1.)
          },
          position if position < (36. / 36.) => {
            let ratio = (position - 25. / 36.) / (11. / 36.);
            (1. - ratio) * na::Vector3::new(-0.8072, -0.4773, 1.) + ratio * na::Vector3::new( 0.7794, -0.4861, 1.)
          },
          _ => unreachable!(),
        }.normalize()
      })
    }
    _ => todo!(),
  })
}

pub fn default_config() -> HashMap<String, HashMap<String, String>> {
  let mut config: HashMap<String, HashMap<String, String>> = HashMap::new();
  let mut config_main: HashMap<String, String> = HashMap::new();
  config_main.insert(String::from("log_level"), String::from("debug"));
  config_main.insert(String::from("loader"), String::from("prophesee"));
  config.insert("main".to_string(), config_main);
  let mut config_ps: HashMap<String, String> = HashMap::new();
  config_ps.insert(String::from("opencl_platform_id"), String::from("0"));
  config_ps.insert(String::from("opencl_device_id"), String::from("0"));
  config_ps.insert(String::from("scan_pattern"), String::from("circle_with_calibration"));
  // Centimeter
  config_ps.insert(String::from("circle_light_diameter"), String::from("15."));
  // Centimeter
  config_ps.insert(String::from("circle_object_distance"), String::from("7."));
  // Centimeter, set to 0. if you don't want near light calibration
  config_ps.insert(String::from("circle_view_width"), String::from("3.8"));
  config_ps.insert(String::from("hypotrochoid_r_big"), String::from("0.3"));
  config_ps.insert(String::from("hypotrochoid_r_small"), String::from("0.7"));
  config_ps.insert(String::from("event_threshold"), String::from("0.20"));
  // Microseconds
  config_ps.insert(String::from("event_refractory"), String::from("300"));
  // Microseconds
  config_ps.insert(String::from("event_refractory_threshold_min"), String::from("512"));
  // Microseconds
  config_ps.insert(String::from("event_refractory_threshold_max"), String::from("2048"));
  // Microseconds
  config_ps.insert(String::from("vis_half_life"), String::from("2048"));
  // Microseconds
  config_ps.insert(String::from("ls_ps_half_life"), String::from("32768"));
  config_ps.insert(String::from("show_ls_ps"), String::from("cv"));
  config_ps.insert(String::from("mask"), String::from(""));
  // There must be an aligned event flush in loader
  config_ps.insert(String::from("ps_fcn_per_n_bin"), String::from("0"));
  config_ps.insert(String::from("ps_fcn_python"), String::from(""));
  // Microseconds, must be power of 2
  config_ps.insert(String::from("record_time"), String::from("65536"));
  config_ps.insert(String::from("record_n_bin"), String::from("16"));
  // There must be an aligned event flush in loader
  config_ps.insert(String::from("cnn_ps_per_n_bin"), String::from("0"));
  config_ps.insert(String::from("cnn_ps_python"), String::from(""));
  // Microseconds
  config_ps.insert(String::from("cnn_ps_half_life"), String::from("32768"));
  config.insert("ps".to_string(), config_ps);
  let mut config_loader_render: HashMap<String, String> = HashMap::new();
  config_loader_render.insert(String::from("client_connect"), String::from("/var/run/libredr_client.sock"));
  config_loader_render.insert(String::from("client_unix"), String::from("true"));
  config_loader_render.insert(String::from("client_tls"), String::from("false"));
  config_loader_render.insert(String::from("obj_file"), String::from(""));
  config_loader_render.insert(String::from("transform_v"),
    String::from("1. 0. 0. 0. | 0. 1. 0. 0. | 0. 0. 1. 0."));
  config_loader_render.insert(String::from("transform_v_frame"),
    String::from("1. 0. 0. 0. | 0. 1. 0. 0. | 0. 0. 1. 0."));
  config_loader_render.insert(String::from("scan_pattern"), String::from("circle"));
  config_loader_render.insert(String::from("circle_latitude"), String::from("45"));
  config_loader_render.insert(String::from("hypotrochoid_r_big"), String::from("0.3"));
  config_loader_render.insert(String::from("hypotrochoid_start_big"), String::from("0."));
  config_loader_render.insert(String::from("hypotrochoid_r_small"), String::from("0.7"));
  config_loader_render.insert(String::from("hypotrochoid_start_small"), String::from("0."));
  config_loader_render.insert(String::from("n_frames"), String::from("100"));
  config_loader_render.insert(String::from("n_rounds"), String::from("1"));
  // Seconds
  config_loader_render.insert(String::from("duration"), String::from("1.0"));
  config_loader_render.insert(String::from("width"), String::from("256"));
  config_loader_render.insert(String::from("height"), String::from("256"));
  config_loader_render.insert(String::from("texture_resolution"), String::from("256"));
  config_loader_render.insert(String::from("diffuse_enable"), String::from("true"));
  config_loader_render.insert(String::from("specular_enable"), String::from("true"));
  config_loader_render.insert(String::from("event_threshold_mean"), String::from("0.1"));
  config_loader_render.insert(String::from("event_threshold_std"), String::from("0.001"));
  config_loader_render.insert(String::from("event_refractory"), String::from("0.000580"));
  config_loader_render.insert(String::from("show_video"), String::from("cv"));
  config_loader_render.insert(String::from("save_video"), String::from(""));
  config_loader_render.insert(String::from("save_normal"), String::from(""));
  config_loader_render.insert(String::from("save_roughness"), String::from(""));
  config_loader_render.insert(String::from("load_video"), String::from(""));
  config_loader_render.insert(String::from("load_normal"), String::from(""));
  config_loader_render.insert(String::from("save_event"), String::from(""));
  config_loader_render.insert(String::from("seed"), String::from("0"));
  config.insert("loader_render".to_string(), config_loader_render);
  let mut config_loader_event_reader: HashMap<String, String> = HashMap::new();
  config_loader_event_reader.insert(String::from("width"), String::from("256"));
  config_loader_event_reader.insert(String::from("height"), String::from("256"));
  config_loader_event_reader.insert(String::from("load_event"), String::from(""));
  config_loader_event_reader.insert(String::from("playback_speed"), String::from("1."));
  // Microseconds
  config_loader_event_reader.insert(String::from("flush_interval"), String::from("2048"));
  config.insert("loader_event_reader".to_string(), config_loader_event_reader);
  let mut config_loader_prophesee: HashMap<String, String> = HashMap::new();
  config_loader_prophesee.insert(String::from("serial"), String::from(""));
  config_loader_prophesee.insert(String::from("sensor_name"), String::from("IMX636"));
  config_loader_prophesee.insert(String::from("begin_row"), String::from("0"));
  config_loader_prophesee.insert(String::from("end_row"), String::from("720"));
  config_loader_prophesee.insert(String::from("begin_col"), String::from("280"));
  config_loader_prophesee.insert(String::from("end_col"), String::from("1000"));
  config_loader_prophesee.insert(String::from("bias_fo"), String::from("0"));
  config_loader_prophesee.insert(String::from("bias_diff_on"), String::from("-20"));
  config_loader_prophesee.insert(String::from("bias_diff_off"), String::from("-20"));
  config_loader_prophesee.insert(String::from("bias_refr"), String::from("-20"));
  config_loader_prophesee.insert(String::from("save_event"), String::from(""));
  config.insert("loader_prophesee".to_string(), config_loader_prophesee);
  config
}
