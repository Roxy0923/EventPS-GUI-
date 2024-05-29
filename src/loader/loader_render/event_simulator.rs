use std::fs::File;
use std::io::Write;
use std::collections::HashMap;
use std::mem::{size_of, take};
use std::slice::from_raw_parts;
use fastrand::Rng;
use tracing::warn;
use ndarray::prelude::*;
use xz2::write::XzEncoder;
use anyhow::{Result, ensure};
use crossbeam_channel::Sender;
use super::super::{EventCD, EV_BUFFER_N_JOBS, EV_BUFFER_SIZE, new_event_writer};

const INTENSITY_EPS: f32 = 3. / 255.;

fn events_sort_by_time(events: ArrayView2::<f32>) -> Array2::<f32> {
  let mut perm: Vec<usize> = (0..events.nrows()).collect();
  perm.sort_unstable_by(|a, b| events[[*a, 3]].partial_cmp(&events[[*b, 3]]).expect("Internal array"));
  let mut events_sorted = Array2::zeros(events.raw_dim());
  for (mut events_sorted, perm) in events_sorted.outer_iter_mut().zip(perm.into_iter()) {
    events_sorted.assign(&events.row(perm));
  }
  events_sorted
}

#[derive(Clone)]
enum EventStatus {
  Refractory(f32),
  Ready(f32, f32),
}

struct EventConfig {
  event_threshold_mean: f32,
  event_threshold_std: f32,
  event_refractory: f32,
  rng: Rng,
}

impl EventConfig {
  fn sample_threshold(&mut self) -> f32 {
    let threshold = self.event_threshold_mean + (2. * self.rng.f32() - 1.) * f32::sqrt(3.) * self.event_threshold_std;
    f32::max(threshold, 1e-6)
  }
}

pub struct EventSimulator {
  event_status: Array2<EventStatus>,
  arr_event_cd: [Box<EventCD>; EV_BUFFER_N_JOBS],
  last_frame_log: Option<(Array2<f32>, f32)>,
  arr_ev_sender: [Sender<Box<EventCD>>; EV_BUFFER_N_JOBS],
  event_writer: Option<(XzEncoder<File>, File)>,
  event_config: EventConfig,
  i_frame: usize,
}

impl EventSimulator {
  pub fn new(
      config_loader_render: &HashMap<String, String>,
      arr_ev_sender: [Sender<Box<EventCD>>; EV_BUFFER_N_JOBS]) -> Result<Self> {
    let width = config_loader_render["width"].parse::<usize>()?;
    let height = config_loader_render["height"].parse::<usize>()?;
    let event_threshold_mean = config_loader_render["event_threshold_mean"].parse::<f32>()?;
    let event_threshold_std = config_loader_render["event_threshold_std"].parse::<f32>()?;
    let event_refractory = config_loader_render["event_refractory"].parse::<f32>()?;
    let rng = Rng::with_seed(config_loader_render["seed"].parse::<u64>()?);
    let event_writer = new_event_writer(&config_loader_render["save_event"], 0)?;
    let event_config = EventConfig {
      event_threshold_mean,
      event_threshold_std,
      event_refractory,
      rng,
    };
    Ok(EventSimulator {
      event_status: Array2::from_elem((height, width), EventStatus::Refractory(event_refractory)),
      arr_event_cd: std::array::from_fn(|_| Box::new(EventCD::new())),
      last_frame_log: None,
      arr_ev_sender,
      event_writer,
      event_config,
      i_frame: 0,
    })
  }

  pub fn add_events(&mut self, events: ArrayView2<f32>) -> Result<()> {
    let events = events_sort_by_time(events);
    if let Some((event_writer, _)) = &mut self.event_writer {
      unsafe {
        event_writer.write_all(from_raw_parts(events.as_ptr() as *const u8, events.len() * size_of::<f32>()))?;
      }
    }
    for event in events.rows() {
      let i_col = event[1] as usize;
      let event_cd = &mut self.arr_event_cd[i_col % EV_BUFFER_N_JOBS];
      let ev_sender = &self.arr_ev_sender[i_col % EV_BUFFER_N_JOBS];
      event_cd.i_row[event_cd.n_ev] = event[0] as u16;
      event_cd.i_col[event_cd.n_ev] = event[1] as u16;
      event_cd.polarity[event_cd.n_ev] = (event[2] > 0.) as u8;
      event_cd.timestamp[event_cd.n_ev] = (event[3] * 1e6) as u32;
      event_cd.n_ev += 1;
      if event_cd.n_ev == EV_BUFFER_SIZE {
        if ev_sender.is_full() {
          warn!("ev channel full!");
        }
        ev_sender.send(take(event_cd)).expect("Receiver Should Not Panic");
      }
    }
    Ok(())
  }

  pub fn add_triggers(&mut self, triggers: ArrayView1<f32>) -> Result<()> {
    if let Some((_, event_writer)) = &mut self.event_writer {
      unsafe {
        event_writer.write_all(from_raw_parts(triggers.as_ptr() as *const u8, triggers.len() * size_of::<f32>()))?;
      }
    }
    Ok(())
  }

  pub fn add_frame(&mut self, frame: ArrayView3<f32>, time: f32) -> Result<()> {
    ensure!(frame.shape()[0] == self.event_status.shape()[0]);
    ensure!(frame.shape()[1] == self.event_status.shape()[1]);
    ensure!(frame.shape()[2] == 3);
    let frame_log = frame.map_axis(Axis(2), |frame| {
      let gray = frame[0] * 0.114 + frame[1] * 0.587 + frame[2] * 0.299;
      assert!(gray.is_finite(), "{:?}", self.i_frame);
      assert!(gray >= 0., "{:?}", self.i_frame);
      assert!(gray < 1e6, "{:?}", self.i_frame);
      (gray + INTENSITY_EPS).ln()
    });
    if let Some((last_frame_log, last_time)) = &self.last_frame_log {
      ensure!(time > last_time);
      let mut events = Array2::zeros((0, 4));
      for i_row in 0..frame_log.nrows() {
        for i_col in 0..frame_log.ncols() {
          let event_status = &mut self.event_status[(i_row, i_col)];
          let frame_log = &frame_log[(i_row, i_col)];
          let last_frame_log = &last_frame_log[(i_row, i_col)];
          loop {
            *event_status = match event_status {
              EventStatus::Refractory(time_ready) => {
                if *time_ready > time {
                  break;
                }
                let ratio = (*time_ready - last_time) / (time - last_time);
                let baseline = last_frame_log * (1. - ratio) + frame_log * ratio;
                let threshold = self.event_config.sample_threshold();
                EventStatus::Ready(baseline, threshold)
              },
              EventStatus::Ready(baseline, threshold) => {
                if (*baseline - frame_log).abs() < *threshold {
                  break;
                }
                let polarity = (frame_log - *baseline).signum();
                let target = *baseline + polarity * *threshold;
                let ratio = (target - last_frame_log) / (frame_log - last_frame_log);
                let trigger_time = last_time * (1. - ratio) + time * ratio;
                events.push_row(aview1(&[i_row as f32, i_col as f32, polarity, trigger_time])).expect("Internal array");
                EventStatus::Refractory(trigger_time + self.event_config.event_refractory)
              },
            }
          }
        }
      }
      self.add_events(events.view())?;
    }
    self.last_frame_log = Some((frame_log, time));
    self.i_frame += 1;
    Ok(())
  }

  pub fn flush_events(&mut self) -> Result<()> {
    for (ev_sender, event_cd) in self.arr_ev_sender.iter().zip(self.arr_event_cd.iter_mut()) {
      ev_sender.send(take(event_cd)).expect("Receiver Should Not Panic");
    }
    Ok(())
  }

  pub fn flush_writer(&mut self) -> Result<()> {
    if let Some((event_writer_a, event_writer_b)) = &mut self.event_writer {
      event_writer_a.flush()?;
      event_writer_b.flush()?;
    }
    Ok(())
  }
}
