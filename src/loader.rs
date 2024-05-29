#[cfg(feature = "loader_render")]
mod loader_render;
#[cfg(feature = "loader_prophesee")]
mod loader_prophesee;
mod loader_event_reader;

use std::thread;
use std::fs::File;
use std::mem::take;
use std::boxed::Box;
#[cfg(feature = "display_cv")]
use std::sync::Mutex;
use std::collections::HashMap;
use tracing::{info, warn};
use xz2::write::XzEncoder;
use crossbeam_channel::Receiver;
#[cfg(feature = "loader_render")]
pub use self::loader_render::Render;
use anyhow::{Result, anyhow, ensure};
#[cfg(feature = "loader_prophesee")]
pub use self::loader_prophesee::Prophesee;
pub use self::loader_event_reader::EventReader;

// Must be the same as `EV_BUFFER_SIZE` in `include/loader_prophesee.h`
const EV_BUFFER_SIZE: usize = 8192;
// Must be the same as `EV_BUFFER_N_JOBS` in `include/loader_prophesee.h`
pub const EV_BUFFER_N_JOBS: usize = 4;
const EV_CHANNEL_SIZE: usize = 128;
const EV_CHUNK_CHANNEL_SIZE: usize = 32;
pub const EV_BUFFER_N_BUCKET: usize = 1024;
pub const EV_BUFFER_BUCKET_SIZE: usize = 128;

pub struct EventCD {
  n_ev: usize,
  i_row: [u16; EV_BUFFER_SIZE],
  i_col: [u16; EV_BUFFER_SIZE],
  polarity: [u8; EV_BUFFER_SIZE],
  timestamp: [u32; EV_BUFFER_SIZE],
}

impl EventCD {
  fn new() -> Self {
    Self::default()
  }

  #[cfg(feature = "loader_prophesee")]
  fn from_slice(i_row: &[u16], i_col: &[u16], polarity: &[u8], timestamp: &[u32]) -> Result<Self> {
    Ok(EventCD {
      n_ev: EV_BUFFER_SIZE,
      i_row: i_row.try_into()?,
      i_col: i_col.try_into()?,
      polarity: polarity.try_into()?,
      timestamp: timestamp.try_into()?,
    })
  }
}

impl Default for EventCD {
  fn default() -> Self {
    EventCD {
      n_ev: 0,
      i_row: [0; EV_BUFFER_SIZE],
      i_col: [0; EV_BUFFER_SIZE],
      polarity: [0; EV_BUFFER_SIZE],
      timestamp: [0; EV_BUFFER_SIZE],
    }
  }
}

pub struct EventChunk {
  // begin external trigger timestamp of a full circle
  pub timestamp_begin: u32,
  // end external trigger timestam[] of a full circle
  pub timestamp_end: u32,
  // max timestamp in this chunk
  pub timestamp_max: u32,
  pub n_ev: [u16; EV_BUFFER_N_BUCKET],
  pub i_row: [[u16; EV_BUFFER_N_BUCKET]; EV_BUFFER_BUCKET_SIZE],
  pub i_col: [[u16; EV_BUFFER_N_BUCKET]; EV_BUFFER_BUCKET_SIZE],
  pub polarity: [[u8; EV_BUFFER_N_BUCKET]; EV_BUFFER_BUCKET_SIZE],
  pub timestamp: [[u32; EV_BUFFER_N_BUCKET]; EV_BUFFER_BUCKET_SIZE],
}

impl EventChunk {
  fn new() -> Self {
    Self::default()
  }
}

impl Default for EventChunk {
  fn default() -> Self {
    EventChunk {
      timestamp_begin: 0,
      timestamp_end: 0,
      timestamp_max: 0,
      n_ev: [0; EV_BUFFER_N_BUCKET],
      i_row: [[0; EV_BUFFER_N_BUCKET]; EV_BUFFER_BUCKET_SIZE],
      i_col: [[0; EV_BUFFER_N_BUCKET]; EV_BUFFER_BUCKET_SIZE],
      polarity: [[0; EV_BUFFER_N_BUCKET]; EV_BUFFER_BUCKET_SIZE],
      timestamp: [[0; EV_BUFFER_N_BUCKET]; EV_BUFFER_BUCKET_SIZE],
    }
  }
}

/// Spawn threads to split events into event chunks
pub fn spawn_split_event(
    begin_row: u16,
    end_row: u16,
    begin_col: u16,
    end_col: u16,
    ev_receiver: Receiver<Box<EventCD>>,
    update: Box<dyn Fn(&EventCD) -> Option<(u32, u32)> + std::marker::Send>,
    ) -> (thread::JoinHandle<()>, Receiver<Box<EventChunk>>) {
  let (ev_chunk_sender, ev_chunk_receiver) = crossbeam_channel::bounded(EV_CHUNK_CHANNEL_SIZE);
  let handle = thread::spawn(move || {
    info!("loader::spawn_split_event: thread started");
    let mut ev_chunk = Box::new(EventChunk::new());
    // Stop thread if `ev_sender` is closed
    let mut timestamp_max = 0;
    while let Ok(ev) = ev_receiver.recv() {
      let Some((timestamp_begin, timestamp_end)) = update(&ev) else {
        return;
      };
      for i_ev in 0..ev.n_ev {
        if !(ev.i_row[i_ev] >= begin_row && ev.i_row[i_ev] < end_row &&
            ev.i_col[i_ev] >= begin_col && ev.i_col[i_ev] < end_col) {
          continue;
        }
        let i_row = ev.i_row[i_ev] - begin_row;
        let i_col = ev.i_col[i_ev] - begin_col;
        let i_bucket = (i_row as usize * (end_col - begin_col) as usize / EV_BUFFER_N_JOBS +
                        i_col as usize / EV_BUFFER_N_JOBS) % EV_BUFFER_N_BUCKET;
        if ev_chunk.n_ev[i_bucket] as usize == EV_BUFFER_BUCKET_SIZE {
          ev_chunk.timestamp_begin = timestamp_begin;
          ev_chunk.timestamp_end = timestamp_end;
          ev_chunk.timestamp_max = timestamp_max;
          ev_chunk_sender.send(take(&mut ev_chunk)).expect("Receiver Should Not Panic");
        }
        let bucket_n_ev = ev_chunk.n_ev[i_bucket] as usize;
        ev_chunk.n_ev[i_bucket] = (bucket_n_ev + 1).try_into().expect("EV_BUFFER_BUCKET_SIZE should be u16");
        ev_chunk.i_row[bucket_n_ev][i_bucket] = i_row;
        ev_chunk.i_col[bucket_n_ev][i_bucket] = i_col;
        ev_chunk.polarity[bucket_n_ev][i_bucket] = ev.polarity[i_ev];
        ev_chunk.timestamp[bucket_n_ev][i_bucket] = ev.timestamp[i_ev];
        timestamp_max = ev.timestamp[i_ev];
      }
      if ev.n_ev != EV_BUFFER_SIZE {
        // flush
        ev_chunk.timestamp_begin = timestamp_begin;
        ev_chunk.timestamp_end = timestamp_end;
        ev_chunk.timestamp_max = timestamp_max;
        ev_chunk_sender.send(take(&mut ev_chunk)).expect("Receiver Should Not Panic");
      }
    }
    info!("loader::spawn_split_event: thread exit");
  });
  (handle, ev_chunk_receiver)
}

pub trait Loader {
  fn new(
      config: &HashMap<String, String>,
      #[cfg(feature = "display_cv")] cv_mutex: &'static Mutex<()>,
      ) -> Result<(Self, [Receiver<Box<EventChunk>>; EV_BUFFER_N_JOBS])>
    where Self: Sized;
  fn get_cd_counter(&self) -> u32;
  fn get_ext_trigger_counter(&self) -> Option<u32>;
}

pub fn new_event_writer(save_event: &str, preset: u32) -> Result<Option<(XzEncoder<File>, File)>> {
  Ok(if save_event.is_empty() {
    None
  } else {
    let save_event: [&str; 2] = save_event
      .split([' ', '|'])
      .filter(|v| !v.is_empty())
      .collect::<Vec<&str>>()
      .try_into()
      .map_err(|_| anyhow!("loader::new_event_writer: \
        `save_event` should be two paths for compressed events and triggers"))?;
    Some((XzEncoder::new(File::create(save_event[0])?, preset), File::create(save_event[1])?))
  })
}
