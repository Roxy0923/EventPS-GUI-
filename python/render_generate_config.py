import os
import math
import hashlib
import configparser
import numpy as np
from scipy.stats import ortho_group

RENDER_RESOLUTION = 256

def generate_config(render_id, dataset, mode, rng):
  seed = rng.integers(65536)
  transform_v = np.zeros((3, 4), dtype=np.float32)
  transform_v[:, :3] = ortho_group.rvs(3, random_state=rng) * rng.uniform(0.5, 1.5)
  if dataset == "blobs":
    if mode == "training":
      scene_id = rng.integers(1, 8 + 1)
    elif mode == "eval":
      scene_id = rng.integers(9, 10 + 1)
    if render_id == 0:
      scene_id = 0
      transform_v[:, :3] = np.eye(3)
  elif dataset == "sculptures":
    if mode == "training":
      scene_id = rng.integers(0, 12 + 1)
    elif mode == "eval":
      scene_id = rng.integers(12, 15 + 1)
  config = configparser.ConfigParser(interpolation=None)
  config["main"] = {
    "log_level": "info",
    "loader": "render",
  }
  scan_pattern = rng.choice(["circle", "hypotrochoid", "diligent"])
  circle_latitude = rng.integers(30, 60 + 1)
  hypotrochoid_r_big = rng.uniform(0.2, 0.3)
  hypotrochoid_r_small = rng.uniform(0.6, 0.7)
  event_threshold_mean = rng.uniform(0.05, 0.25)
  event_threshold_std = rng.uniform(0.05, 0.35) * event_threshold_mean
  event_threshold = rng.uniform(0.80, 1.20) * event_threshold_mean
  event_refractory = rng.integers(300, 2500) # us
  duration = rng.uniform(0.2, 0.4) # 6 / 30 seconds to 12 / 30 seconds
  texture_resolution = rng.choice([16, 32, 64])
  config["ps"] = {
    "scan_pattern": "circle_with_calibration" if scan_pattern == "circle" else scan_pattern,
    "circle_light_diameter": 2 * math.cos(circle_latitude / 180 * math.pi),
    "circle_object_distance": math.sin(circle_latitude / 180 * math.pi),
    "circle_view_width": 0.,
    "hypotrochoid_r_big": hypotrochoid_r_big,
    "hypotrochoid_r_small": hypotrochoid_r_small,
    "event_threshold": event_threshold,
    "event_refractory": event_refractory,
    "event_refractory_threshold_min": event_refractory,
    "event_refractory_threshold_max": 8192,      # us
    "record_time": 131072,                       # us, about 1-2 rounds
    "record_n_bin": 16,
    "vis_half_life": 2048,                       # us, about 1/16 rounds
    "ls_ps_half_life": 32768,                    # us, about one round
    "show_ls_ps": "none",
    "ps_fcn_per_n_bin": 0,
    "ps_fcn_python": "python/ps_fcn_eval.py",
    "cnn_ps_per_n_bin": 0,
    "cnn_ps_python": "python/cnn_ps_eval.py",
    "cnn_ps_half_life": 32768,                   # us, about one round
  }
  config["loader_render"] = {
    "obj_file": f"data/{dataset}_processed/{scene_id:06}.obj",
    "client_connect": "/tmp/libredr_client.sock",
    "client_unix": "true",
    "client_tls": "false",
    # Transform for the beginning
    "transform_v": " ".join(map(str, transform_v.flatten().tolist())),
    # Apply this transform to the left of transform_v per frame
    "transform_v_frame": "1. 0. 0. 0. | 0. 1. 0. 0. | 0. 0. 1. 0.",
    "scan_pattern": scan_pattern,
    "circle_latitude": circle_latitude,
    "hypotrochoid_r_big": hypotrochoid_r_big,
    "hypotrochoid_start_big": 0.,
    "hypotrochoid_r_small": hypotrochoid_r_small,
    "hypotrochoid_start_small": 0.,
    "n_frames": 600,
    "n_rounds": 6,
    "duration": duration,
    "width": RENDER_RESOLUTION,
    "height": RENDER_RESOLUTION,
    "texture_resolution": texture_resolution,
    "specular_enable": "true",
    "event_threshold_mean": event_threshold_mean,
    "event_threshold_std": event_threshold_std,
    "event_refractory": event_refractory * 1e-6,
    "show_video": "cv",
    # "load_video": f"data/{dataset}_{mode}/{render_id:06}/frames.vif data/{dataset}_{mode}/{render_id:06}/normal.vif",
    "save_video": f"data/{dataset}_{mode}/{render_id:06}/frames.vif data/{dataset}_{mode}/{render_id:06}/normal.vif",
    "save_normal": f"data/{dataset}_{mode}/{render_id:06}/normal.xz",
    "save_event": f"data/{dataset}_{mode}/{render_id:06}/event_internal.xz " + \
                  f"data/{dataset}_{mode}/{render_id:06}/event_trigger",
    "seed": seed,
  }
  config["loader_event_reader"] = {
    "width": RENDER_RESOLUTION,
    "height": RENDER_RESOLUTION,
    "load_event": f"data/{dataset}_{mode}/{render_id:06}/event.xz data/{dataset}_{mode}/{render_id:06}/event_trigger",
    "playback_speed": 0.,
    "flush_interval": 2048, # us
  }
  os.makedirs(f"data/{dataset}_{mode}/{render_id:06}/", exist_ok=True)
  with open(f"data/{dataset}_{mode}/{render_id:06}/render.ini", "w") as f:
    config.write(f)
  config["main"]["loader"] = "event_reader"
  config["loader_render"]["load_normal"] = config["loader_render"]["save_normal"]
  config_ps_fcn = configparser.ConfigParser(interpolation=None)
  config_ps_fcn.read_dict(config)
  config_ps_fcn["ps"]["ps_fcn_per_n_bin"] = "1"
  with open(f"data/{dataset}_{mode}/{render_id:06}/eval_ps_fcn.ini", "w") as f:
    config_ps_fcn.write(f)
  config_cnn_ps = configparser.ConfigParser(interpolation=None)
  config_cnn_ps.read_dict(config)
  config_cnn_ps["ps"]["cnn_ps_per_n_bin"] = "1"
  with open(f"data/{dataset}_{mode}/{render_id:06}/eval_cnn_ps.ini", "w") as f:
    config_cnn_ps.write(f)

def main():
  for dataset, mode, n_render in [("blobs",      "training", 20),
                                  ("blobs",      "eval",     20),
                                  ("sculptures", "training", 100),
                                  ("sculptures", "eval",     20)]:
    for render_id in range(n_render):
      seed = int.from_bytes(hashlib.md5(str((dataset, mode, render_id)).encode()).digest()[:4])
      # if render_id == 0:
      #   print(seed)
      rng = np.random.default_rng(seed=seed)
      config = generate_config(render_id, dataset, mode, rng)

if __name__ == "__main__":
  main()
