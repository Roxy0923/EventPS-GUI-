
use std::thread;
use std::fs::File;
use std::io::Read;
use std::marker::Send;
use std::mem::size_of;
use std::io::BufReader;
use std::sync::{Arc, Weak, Mutex};
use std::collections::VecDeque;
use std::time::{Instant, Duration};
use std::slice::{from_raw_parts, from_ref, from_mut};
use tracing::error;
use xz2::read::XzDecoder;
use anyhow::{anyhow, bail};
use crossbeam_channel::{Sender, Receiver};
use super::*;

struct EventReaderData {
  cd_counter: u32,
  triggers: VecDeque<f32>,
}

impl EventReaderData {
  fn new() -> Self {
    EventReaderData {
      cd_counter: 0,
      triggers: VecDeque::new(),
    }
  }
}

pub struct EventReader {
  data: Arc<Mutex<EventReaderData>>,
}

fn flush_events(
    event_reader_data_weak: &Weak<Mutex<EventReaderData>>,
    arr_event_cd: &mut[Box<EventCD>],
    arr_ev_sender: &[Sender<Box<EventCD>>],
    timestamp_max: f32,
    show_warn: bool) -> Result<()> {
  {
    // Stop thread if `event_reader_data` is dropped
    let Some(event_reader_data) = event_reader_data_weak.upgrade() else {
      bail!("`event_reader_data` is dropped");
    };
    let mut event_reader_data = event_reader_data.lock().expect("No task should panic");
    while event_reader_data.triggers.len() > 2 && timestamp_max > event_reader_data.triggers[0] {
      event_reader_data.triggers.pop_front();
    }
  }
  for (event_cd, ev_sender) in arr_event_cd.iter_mut().zip(arr_ev_sender.iter()) {
    if show_warn && ev_sender.is_full() {
      warn!("ev channel full!");
    }
    ev_sender.send(take(event_cd)).expect("Receiver Should Not Panic");
  }
  Ok(())
}

fn spawn_reader<T: Read + Send + 'static>(
    config_loader_event_loader: &HashMap<String, String>,
    event_reader_data_weak: Weak<Mutex<EventReaderData>>,
    mut reader: T,
    arr_ev_sender: [Sender<Box<EventCD>>; EV_BUFFER_N_JOBS]) -> Result<()> {
  let height = config_loader_event_loader["height"].parse::<u16>()?;
  let width = config_loader_event_loader["width"].parse::<u16>()?;
  let playback_speed = config_loader_event_loader["playback_speed"].parse::<f32>()?;
  ensure!(playback_speed >= 0.);
  let show_warn = playback_speed > 0.;
  let flush_interval = config_loader_event_loader["flush_interval"].parse::<u32>()? as f32 * 1e-6;
  let mut i_flush = 0;
  thread::spawn(move || {
    let ret = (move || {
      let mut arr_event_cd:[Box<EventCD>; EV_BUFFER_N_JOBS] = std::array::from_fn(|_| Box::new(EventCD::new()));
      let instant_start = Instant::now();
      loop {
        let mut read_buf = [0; 4 * size_of::<f32>()];
        if let Err(e) = reader.read_exact(&mut read_buf) {
          if e.kind() == std::io::ErrorKind::UnexpectedEof {
            break;
          }
          bail!("{e:?}");
        }
        let event = unsafe { from_raw_parts(read_buf.as_ptr() as *const f32, 4) };
        if flush_interval > 0. && event[3] >= flush_interval * i_flush as f32 {
          flush_events(&event_reader_data_weak, &mut arr_event_cd, &arr_ev_sender, event[3], show_warn)?;
          i_flush += 1;
        }
        let i_col = event[1] as usize;
        let event_cd = &mut arr_event_cd[i_col % EV_BUFFER_N_JOBS];
        let ev_sender = &arr_ev_sender[i_col % EV_BUFFER_N_JOBS];
        event_cd.i_row[event_cd.n_ev] = event[0] as u16;
        ensure!((event[0] as u16) < height);
        event_cd.i_col[event_cd.n_ev] = event[1] as u16;
        ensure!((event[1] as u16) < width);
        event_cd.polarity[event_cd.n_ev] = (event[2] > 0.) as u8;
        event_cd.timestamp[event_cd.n_ev] = (event[3] * 1e6) as u32;
        event_cd.n_ev += 1;
        while playback_speed > 0. && event[3] > instant_start.elapsed().as_secs_f32() * playback_speed {
          thread::sleep(Duration::from_millis(10));
        }
        if event_cd.n_ev == EV_BUFFER_SIZE {
          flush_events(&event_reader_data_weak, from_mut(event_cd), from_ref(ev_sender), event[3], show_warn)?;
        }
      }
      Ok(String::from("Finish reading events!"))
    })();
    if let Err(e) = ret {
      error!("EventReader::spawn_reader: {e:?}");
      panic!("EventReader::spawn_reader: {e:?}");
    }
    info!("EventReader::spawn_reader: thread exit {ret:?}");
  });
  Ok(())
}

impl Loader for EventReader {
  fn new(
      config_loader_event_loader: &HashMap<String, String>,
      #[cfg(feature = "display_cv")] _cv_mutex: &'static Mutex<()>,
      ) -> Result<(Self, [Receiver<Box<EventChunk>>; EV_BUFFER_N_JOBS])> {
    let height = config_loader_event_loader["height"].parse::<u16>()?;
    let width = config_loader_event_loader["width"].parse::<u16>()?;
    let load_event: [&str; 2] = config_loader_event_loader["load_event"]
        .split([' ', '|'])
        .filter(|v| !v.is_empty())
        .collect::<Vec<&str>>()
        .try_into()
        .map_err(|_| anyhow!("EventReader::new: `load_event` should be two paths for compressed events and triggers"))?;
    let mut event_reader_data = EventReaderData::new();
    let triggers = std::fs::read(load_event[1])?;
    ensure!(triggers.len() % size_of::<f32>() == 0);
    ensure!(triggers.len() / size_of::<f32>() >= 2);
    event_reader_data.triggers = VecDeque::from(unsafe {
      from_raw_parts(triggers.as_ptr() as *const f32, triggers.len() / size_of::<f32>()).to_vec()
    });
    let event_reader_data = Arc::new(Mutex::new(event_reader_data));
    let (vec_ev_sender, vec_ev_chunk_receiver) = (0..EV_BUFFER_N_JOBS).map(|_| {
      let event_reader_data_weak = Arc::downgrade(&event_reader_data);
      let (ev_sender, ev_receiver) = crossbeam_channel::bounded(EV_CHANNEL_SIZE);
      let update = Box::new(move |ev: &EventCD| {
        // Stop thread if `event_reader_data` is dropped
        let Some(event_reader_data) = event_reader_data_weak.upgrade() else {
          return None;
        };
        let mut event_reader_data = event_reader_data.lock().expect("No task should panic");
        event_reader_data.cd_counter += ev.n_ev as u32;
        Some(((event_reader_data.triggers[0] * 1e6) as u32,
              (event_reader_data.triggers[1] * 1e6) as u32))
      });
      let (_, ev_chunk_receiver) = spawn_split_event(0, height, 0, width, ev_receiver, update);
      (ev_sender, ev_chunk_receiver)
    }).unzip::<_, _, Vec<_>, Vec<_>>();
    let reader = BufReader::new(XzDecoder::new(File::open(load_event[0])?));
    spawn_reader(config_loader_event_loader,
                 Arc::downgrade(&event_reader_data),
                 reader,
                 vec_ev_sender.try_into().expect("EV_BUFFER_N_JOBS"))?;
    let camera = EventReader {
      data: event_reader_data,
    };
    Ok((camera, vec_ev_chunk_receiver.try_into().expect("EV_BUFFER_N_JOBS")))
  }

  fn get_cd_counter(&self) -> u32 {
    let data = self.data.lock().expect("No task should panic");
    data.cd_counter
  }

  fn get_ext_trigger_counter(&self) -> Option<u32> {
    None
  }
}
