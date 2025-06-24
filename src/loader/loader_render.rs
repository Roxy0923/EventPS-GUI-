mod render_frame;
mod video_writer;
pub mod event_simulator;

use std::io::Write;
use std::boxed::Box;
use std::path::Path;
#[cfg(feature = "display_cv")]
use std::sync::Mutex;
use std::mem::size_of;
use std::slice::from_raw_parts;
use std::collections::{VecDeque, HashMap};
use std::sync::{Arc, atomic::{AtomicU32, Ordering}};
#[cfg(feature = "display_cv")]
use cv_convert::TryToCv;
use ndarray::prelude::*;
use tracing::{info, error};
use anyhow::{Result, bail};
use crossbeam_channel::{Sender, Receiver};
use tokio::{runtime::Runtime, task::JoinHandle};
#[cfg(feature = "display_cv")]
use opencv::{prelude::*, core::CV_8UC3, videoio};
use libredr::Geometry;
use super::*;
use self::video_writer::VideoWriter;
use self::render_frame::render_frame;
use self::event_simulator::EventSimulator;

pub struct Render {
  cd_counter: Arc<AtomicU32>,
  _runtime: Runtime,
  _task_render: JoinHandle<()>,
}

fn parse_matrix_4(config: &str) -> Result<Array2<f32>> {
  let mut matrix_value = config.split([' ', '|'])
    .filter(|v| !v.is_empty())
    .map(str::parse::<f32>)
    .collect::<std::result::Result<Vec<f32>, std::num::ParseFloatError>>()?;
  matrix_value.extend_from_slice(&[0., 0., 0., 1.]);
  Ok(Array2::from_shape_vec((4, 4), matrix_value)?)
}

async fn render_events(
    config_loader_render: HashMap<String, String>,
    arr_ev_sender: [Sender<Box<EventCD>>; EV_BUFFER_N_JOBS],
    #[cfg(feature = "display_cv")] cv_mutex: &'static Mutex<()>,
    ) -> Result<String> {
  let mut i_queue_frame = 0;
  let data_cache = Default::default();
  let mut queue = VecDeque::new();
  let config_loader_render = Arc::new(config_loader_render);
  let mut event_simulator = EventSimulator::new(&config_loader_render, arr_ev_sender)?;
  let n_rounds = config_loader_render["n_rounds"].parse::<usize>()?;
  let duration = config_loader_render["duration"].parse::<f32>()?;
  // Must contain only the first two triggers. Because hypotrochoid is not periodic!
  // The real time is interpolated by the first two triggers.
  event_simulator.add_triggers(Array1::linspace(0., duration / n_rounds as f32, 2).view())?;
  let mut video_writer = if config_loader_render["save_video"].is_empty() {
    None
  } else {
    Some(VideoWriter::new(&config_loader_render)?)
  };
  let mut transform_v = parse_matrix_4(&config_loader_render["transform_v"])?;
  let transform_v_frame = parse_matrix_4(&config_loader_render["transform_v_frame"])?;
  let n_frames = config_loader_render["n_frames"].parse::<usize>()?;
  let save_normal = config_loader_render["save_normal"].as_str();
  let save_roughness = config_loader_render["save_roughness"].as_str();
  let load_normal = config_loader_render["load_normal"].as_str();
  let n_rows = config_loader_render["height"].parse::<usize>()?;
  let n_cols = config_loader_render["width"].parse::<usize>()?;
  let load_video = config_loader_render["load_video"].as_str();
  let load_video_check_path = if load_video.contains("%") {
    Path::new(load_video).parent().ok_or(anyhow!("`load_video` contains % but can't determine parent folder"))?
  } else {
    Path::new(load_video)
  };
  let (load_video, vertex_hash) = if load_video_check_path.try_exists()? {
    #[cfg(feature = "display_cv")]
    {
      let video_capture = videoio::VideoCapture::from_file(load_video, videoio::CAP_FFMPEG)?;
      let render_normal = if load_normal.is_empty() {
        Array3::zeros((3, n_rows, n_cols))
      } else {
        crate::load_normal(load_normal, n_rows, n_cols)?
      };
      (Some((Mutex::new(video_capture), render_normal)), Default::default())
    }
    #[cfg(not(feature = "display_cv"))]
    {
      bail!("render_events: `display_cv` feature is not enabled in this compile, required by `load_video`");
    }
  } else {
    let obj_file = Path::new(&config_loader_render["obj_file"]);
    let mut geometry = Geometry::new();
    let vertex_hash = geometry.add_obj(obj_file, Array2::eye(4), Array2::eye(3), &data_cache)?;
    #[cfg(feature = "display_cv")]
    let ret = (None::<(Mutex<videoio::VideoCapture>, Array3<f32>)>, vertex_hash);
    #[cfg(not(feature = "display_cv"))]
    let ret = (None::<((), ())>, vertex_hash);
    ret
  };
  for i_frame in 0..n_frames {
    let (render_image, render_normal) = if let Some((video_capture, render_normal)) = &load_video {
      #[cfg(feature = "display_cv")]
      {
        let mut video_capture = video_capture.lock().unwrap();
        let mut render_image = Mat::new_rows_cols_with_default(n_rows.try_into()?,
                                                              n_cols.try_into()?,
                                                              CV_8UC3,
                                                              Default::default())?;
        ensure!(video_capture.read(&mut render_image)?);
        let render_image:ArrayBase<_, _> = render_image.try_to_cv()?;
        let render_image = render_image.map(|x: &u8| (*x as f32 / 255.).powf(2.2));
        (render_image.permuted_axes([2, 0, 1]), render_normal.to_owned())
      }
      #[cfg(not(feature = "display_cv"))]
      {
        unreachable!();
      }
    } else {
      while i_queue_frame < n_frames && i_queue_frame < i_frame + 2 * EV_BUFFER_N_JOBS {
        let vertex_hash = vertex_hash.to_owned();
        let transform_v_clone = transform_v.to_owned();
        let data_cache = data_cache.to_owned();
        let config_loader_render = config_loader_render.clone();
        let future_render_frame = async move {
          render_frame(i_queue_frame, &config_loader_render, vertex_hash, transform_v_clone, data_cache).await
        };
        queue.push_back(tokio::spawn(future_render_frame));
        transform_v = transform_v_frame.dot(&transform_v);
        i_queue_frame += 1;
      }
      let (render, render_roughness) = queue.pop_front().expect("internal").await??;
      let render_image = render.slice(s![0..3, .., ..]).as_standard_layout().into_owned();
      let render_normal = render.slice(s![6..9, .., ..]).as_standard_layout().into_owned();
      if i_frame == 0 && !save_roughness.is_empty() {
        let render_roughness = render_roughness.as_standard_layout().into_owned();
        let mut writer = xz2::write::XzEncoder::new(std::fs::File::create(save_roughness)?, 6);
        unsafe {
          let data = from_raw_parts(render_roughness.as_ptr() as *const u8, render_roughness.len() * size_of::<f32>());
          writer.write_all(data)?;
        }
      }
      (render_image, render_normal)
    };
    if i_frame == 0 && !save_normal.is_empty() {
      let mut writer = xz2::write::XzEncoder::new(std::fs::File::create(save_normal)?, 6);
      unsafe {
        writer.write_all(from_raw_parts(render_normal.as_ptr() as *const u8, render_normal.len() * size_of::<f32>()))?;
      }
    }
    match config_loader_render["show_video"].as_str() {
      "none" => (),
      "cv" => {
        #[cfg(feature = "display_cv")]
        {
          crate::cv_show_image("render_image", render_image.view(), cv_mutex)?;
          crate::cv_show_normal("render_normal", render_normal.view(), cv_mutex)?;
        }
        #[cfg(not(feature = "display_cv"))]
        {
          bail!("render_events: `display_cv` feature is not enabled in this compile");
        }
      },
      others => {
        bail!("render_events: Unexpected show_video: {others}");
      }
    }
    if let Some(video_writer) = &mut video_writer {
      video_writer.add_frame(render_image.view(), render_normal.view())?;
    }
    event_simulator.add_frame(render_image.view().permuted_axes([1, 2, 0]),
                              i_frame as f32 / (n_frames - 1) as f32 * duration)?;
    event_simulator.flush_events()?;
  }
  if let Some(video_writer) = &mut video_writer {
    video_writer.flush()?;
  }
  event_simulator.flush_writer()?;
  Ok(String::from("Finish rendering all frames!"))
}

impl Loader for Render {
  fn new(
      config_loader_render: &HashMap<String, String>,
      #[cfg(feature = "display_cv")] cv_mutex: &'static Mutex<()>,
      ) -> Result<(Self, [Receiver<Box<EventChunk>>; EV_BUFFER_N_JOBS])> {
    let n_rounds = config_loader_render["n_rounds"].parse::<usize>()?;
    let duration = config_loader_render["duration"].parse::<f32>()?;
    let height = config_loader_render["height"].parse::<u16>()?;
    let width = config_loader_render["width"].parse::<u16>()?;
    let cd_counter = Arc::new(AtomicU32::new(0));
    let runtime = tokio::runtime::Builder::new_multi_thread().worker_threads(1).enable_all().build()?;
    let (vec_ev_sender, vec_ev_chunk_receiver) = (0..EV_BUFFER_N_JOBS).map(|_| {
      let cd_counter_weak = Arc::downgrade(&cd_counter);
      let (ev_sender, ev_receiver) = crossbeam_channel::bounded(EV_CHANNEL_SIZE);
      let update = Box::new(move |ev: &EventCD| {
        // Stop thread if `cd_counter` is dropped
        let Some(cd_counter) = cd_counter_weak.upgrade() else {
          return None;
        };
        cd_counter.fetch_add(ev.n_ev as u32, Ordering::Relaxed);
        Some((0, (duration / n_rounds as f32 * 1e6) as u32))
      });
      let (_, ev_chunk_receiver) = spawn_split_event(0, height, 0, width, ev_receiver, update);
      (ev_sender, ev_chunk_receiver)
    }).unzip::<_, _, Vec<_>, Vec<_>>();
    let config_loader_render = config_loader_render.to_owned();
    let task_render = runtime.spawn(async move {
      let arr_ev_sender = vec_ev_sender.try_into().expect("EV_BUFFER_N_JOBS");
      let ret = render_events(config_loader_render,
                              arr_ev_sender,
                              #[cfg(feature = "display_cv")] cv_mutex).await;
      if let Err(e) = ret {
        error!("Render::render_events: {e:?}");
        panic!("Render::render_events: {e:?}");
      }
      info!("Render::render_events: thread exit {ret:?}");
    });
    let camera = Render {
      cd_counter,
      _runtime: runtime,
      _task_render: task_render,
    };
    Ok((camera, vec_ev_chunk_receiver.try_into().expect("EV_BUFFER_N_JOBS")))
  }

  fn get_cd_counter(&self) -> u32 {
    self.cd_counter.load(Ordering::Relaxed)
  }

  fn get_ext_trigger_counter(&self) -> Option<u32> {
    None
  }
}
