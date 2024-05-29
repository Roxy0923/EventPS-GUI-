import os
import sys
import lzma
import time
import shutil
import subprocess
import configparser
import multiprocessing
import h5py
import numpy as np

TMP_PATH = "/tmp/v2e/"

def process(data_path):
  print(time.time(), "process", data_path)
  tmp_path = os.path.join(TMP_PATH, str(os.getpid()))
  config = configparser.ConfigParser(interpolation=None)
  config.read(os.path.join(data_path, "render.ini"))
  if "load_video" in config["loader_render"]:
    load_video = config["loader_render"]["load_video"].split(" ")[0]
  else:
    load_video = config["loader_render"]["save_video"].split(" ")[0]
  shutil.rmtree(tmp_path, ignore_errors = True)
  input_path = os.path.join(tmp_path, "input")
  output_path = os.path.join(tmp_path, "output")
  os.makedirs(input_path)
  os.makedirs(output_path)
  if os.path.isdir(load_video):
    input_path = load_video
  else:
    subprocess.run(["ffmpeg", "-i", load_video, os.path.join(input_path, "%06d.png")])
    # stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    # subprocess.run(["ffmpeg", "-i", load_video, "-vf", "eq=gamma=1/2.2", os.path.join(input_path, "%06d.png")])
    # subprocess.run(["ffmpeg", "-i", load_video, os.path.join(input_path, "%06d.hdr")])
  frame_rate = int(config["loader_render"]["n_frames"]) / float(config["loader_render"]["duration"])
  width = int(config["loader_render"]["width"])
  height = int(config["loader_render"]["height"])
  seed = int(config["loader_render"]["seed"]) + 1
  event_threshold_mean = float(config["loader_render"]["event_threshold_mean"])
  event_threshold_std = float(config["loader_render"]["event_threshold_std"])
  event_refractory = float(config["loader_render"]["event_refractory"])
  # TODO: Use HDR feature of v2e instead of modifying v2e `lin_log`` code
  if event_threshold_std > 0:
    noise_args = [
      f"--sigma_thres={event_threshold_std}",
      "--leak_rate_hz=0.1",
      # "--leak_rate_hz=0",
      "--shot_noise_rate_hz=5.0",
      # "--shot_noise_rate_hz=0",
      "--leak_jitter_fraction=0.1",
      # "--leak_jitter_fraction=0",
      "--noise_rate_cov_decades=0.1",
      # "--noise_rate_cov_decades=0",
    ]
  else:
    noise_args = [
      "--dvs_params=clean"
    ]
  subprocess.run(["v2e.py",
    "--disable_slomo",
    # "--slomo_model=data/SuperSloMo39.ckpt",
    f"--dvs_emulator_seed={seed}",
    f"--input={input_path}",
    f"--input_frame_rate={frame_rate}",
    # Bug in V2E refractory_period:
    # https://github.com/SensorsINI/v2e/blob/e960a343d32a5d86b8cd1f84aaa59e7e76bd45ee/v2ecore/emulator.py#L826
    f"--refractory_period={event_refractory}",
    # "--refractory_period=0",
    # `cutoff_hz` doesn't work without slomo
    "--cutoff_hz=0",
    # "--cutoff_hz=200",
    f"--pos_thres={event_threshold_mean}",
    f"--neg_thres={event_threshold_mean}",
    f"--output_folder={output_path}",
    "--unique_output_folder=true",
    f"--output_width={width}",
    f"--output_height={height}",
    "--dvs_h5=events.h5"] + noise_args)
  # stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
  with h5py.File(os.path.join(output_path, "events.h5"), "r") as f:
    events = f.get("events")[...].astype(np.float32)
  events[:, 0] *= 1e-6
  # print("events", events.shape, events.dtype)
  # for i in range(4):
  #   print(f"events[:, {i}]", events[:, i].min(), events[:, i].max(), events[:, i].mean())
  events = np.stack([events[:, 2], events[:, 1], events[:, 3] * 2 - 1, events[:, 0]], axis=-1).tobytes()
  save_event_v2e = os.path.join(data_path, "event_v2e.xz")
  with lzma.LZMAFile(save_event_v2e, "w") as f:
    assert(f.write(events) == len(events))
  load_event = config["loader_event_reader"]["load_event"].split(" ")[0]
  if os.path.isfile(load_event) or os.path.islink(load_event):
    os.remove(load_event)
  os.symlink("event_v2e.xz", load_event)
  # Must contain only the first two. Because hypotrochoid is not periodic
  duration = float(config["loader_render"]["duration"])
  n_rounds = int(config["loader_render"]["n_rounds"])
  event_trigger = np.array([0., duration / n_rounds], dtype=np.float32)
  load_event_trigger = config["loader_event_reader"]["load_event"].split(" ")[1]
  with open(load_event_trigger, "wb") as f:
    f.write(event_trigger.tobytes())
  shutil.rmtree(tmp_path)

def main():
  dataset_path = sys.argv[1]
  pool = multiprocessing.Pool(4)
  data_path = sorted(os.listdir(dataset_path))
  data_path = map(lambda data_path: os.path.join(dataset_path, data_path), data_path)
  pool.map(process, data_path)
  shutil.rmtree(TMP_PATH)

if __name__ == "__main__":
  main()

