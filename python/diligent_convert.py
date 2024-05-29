import io
import os
import sys
import glob
import lzma
import zipfile
import configparser
import numpy as np
import cv2 as cv
from scipy.io import loadmat
from scipy.ndimage import minimum_filter

RENDER_RESOLUTION = 256
BOUNDARY_IDS = [88, 89, 90, 91, 92, 93, 94, 95, 87, 79, 71, 63, 55, 7, 15, 23, 31, 39, 47, 46, 45, 44, 43, 42, 41, 40, 32, 24, 16, 8, 0, 48, 56, 64, 72, 80]
BOUNDARY_IDS = BOUNDARY_IDS * 6 + BOUNDARY_IDS[:1]
CROP_MARGIN = 16

def get_crop_slice(mask):
  up = 0
  while np.all(mask[up, :] == 1):
    up += 1
  down = mask.shape[0] - 1
  while np.all(mask[down, :] == 1):
    down -= 1
  left = 0
  while np.all(mask[:, left] == 1):
    left += 1
  right = mask.shape[1] - 1
  while np.all(mask[:, right] == 1):
    right -= 1
  while right - left < down - up:
    if left > 0:
      left -= 1
    if right - left < down - up and right < mask.shape[1]:
      right += 1
  while down - up < right - left:
    if up > 0:
      up -= 1
    if down - up < right - left and down < mask.shape[0]:
      down += 1
  print("up", up, "down", down, "height", down - up + 1, "left", left, "right", right, "width", right - left + 1)
  return slice(up - CROP_MARGIN, down + 1 + CROP_MARGIN), slice(left - CROP_MARGIN, right + 1 + CROP_MARGIN)

def generate_config(output_dir):
  os.makedirs(output_dir, exist_ok=True)
  config = configparser.ConfigParser(interpolation=None)
  config["main"] = {
    "log_level": "info",
    "loader": "render",
  }
  config["ps"] = {
    "scan_pattern": "diligent",
    "event_threshold": 0.15,
    "event_refractory": 2000,                    # us
    "event_refractory_threshold_min": 2000,      # us
    "event_refractory_threshold_max": 65536,     # us
    "record_time": 131072,                       # us, about 1-2 rounds
    "record_n_bin": 32,                          # about two rounds
    "vis_half_life": 2048,                       # us, about 1/16 rounds
    "ls_ps_half_life": 524288,                   # us, about 1-2 round
    "show_ls_ps": "none",
    "ps_fcn_per_n_bin": 0,
    "ps_fcn_python": "python/ps_fcn_eval.py",
    "cnn_ps_per_n_bin": 0,
    "cnn_ps_python": "python/cnn_ps_eval.py",
    "cnn_ps_half_life": 65536,                   # us, about one round
  }
  config["loader_render"] = {
    "scan_pattern": "diligent",
    "n_frames": len(BOUNDARY_IDS),
    "n_rounds": 6,
    "duration": 0.2,                             # 6 / 30 seconds
    "width": RENDER_RESOLUTION,
    "height": RENDER_RESOLUTION,
    "event_threshold_mean": 0.15,
    "event_threshold_std": 0.0,
    "event_refractory": "0.002000",              # 2000 us
    "show_video": "cv",
    "load_video": os.path.join(output_dir, "frames/%06d.png"),
    "load_normal": os.path.join(output_dir, "normal.xz"),
    "save_event": os.path.join(output_dir, "event_internal.xz") + " " + os.path.join(output_dir, "event_trigger"),
    "seed": 2,
  }
  config["loader_event_reader"] = {
    "width": RENDER_RESOLUTION,
    "height": RENDER_RESOLUTION,
    "load_event": os.path.join(output_dir, "event.xz") + " " + os.path.join(output_dir, "event_trigger"),
    "playback_speed": 0.,
    "flush_interval": 2048, # us
  }
  with open(os.path.join(output_dir, "render.ini"), "w") as f:
    config.write(f)
  config["main"]["loader"] = "event_reader"
  config["ps"]["show_ls_ps"] = "none"
  config_ps = configparser.ConfigParser(interpolation=None)
  config_ps.read_dict(config)
  config_ps["ps"]["ps_fcn_per_n_bin"] = "1"
  config_ps["ps"]["cnn_ps_per_n_bin"] = "1"
  with open(os.path.join(output_dir, "eval.ini"), "w") as f:
    config_ps.write(f)

def process(input_dir, output_dir):
  print(f"input_dir {input_dir}")
  if "bearPNG" in input_dir.name:
    # images = images[20:]
    # light_dirs = light_dirs[20:]
    # not enough pictures on the boundary
    return
  generate_config(output_dir)
  light_intensities = input_dir.joinpath("light_intensities.txt").read_bytes().decode('utf-8').strip().split("\n")
  light_intensities = np.loadtxt(light_intensities)
  light_dirs = input_dir.joinpath("light_directions.txt").read_bytes().decode('utf-8').strip().split("\n")
  light_dirs = np.loadtxt(light_dirs)
  light_dirs_img = np.zeros([1024 ,1024, 3], dtype=np.float32)
  cv.putText(light_dirs_img, "##", (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.8, (1, 1, 1))
  for (i, light_dir) in enumerate(light_dirs):
    print(light_dir)
    light_dir /= light_dir[2]
    cv.putText(light_dirs_img, str(i), (int(512 + 512 * light_dir[0]), int(512 + 512 * light_dir[1])), cv.FONT_HERSHEY_SIMPLEX, 0.8, (1, 1, 1))
  cv.imshow("light_dirs_img", light_dirs_img)
  cv.pollKey()
  images_orig = []
  mask = np.frombuffer(input_dir.joinpath("mask.png").read_bytes(), dtype=np.uint8)
  mask = cv.imdecode(mask, cv.IMREAD_UNCHANGED) == 0
  print("mask", mask.shape, mask.dtype)
  if len(mask.shape) == 3:
    mask = mask[:, :, 0]
  # mask = normal[:, :, 2] <= 0.0
  slice_x, slice_y = get_crop_slice(mask)
  assert(input_dir.joinpath("normal.txt").exists())
  normal = input_dir.joinpath("normal.txt").read_bytes().decode('utf-8').strip().split("\n")
  normal = np.loadtxt(normal).reshape(mask.shape[0], mask.shape[1], 3)
  if "harvestPNG" in input_dir.name:
    normal = normal[::-1, :, :]
  cv.imshow("normal_orig", normal * 0.5 + 0.5)
  normal = normal[:, :, [1, 0, 2]]
  normal[:, :, 0] *= -1
  normal = normal[slice_x, slice_y, :]
  mask = mask[slice_x, slice_y]
  filenames = input_dir.joinpath("filenames.txt").read_bytes().decode('utf-8').strip().split("\n")
  # print(filenames)
  for filename in filenames:
    image_file = np.frombuffer(input_dir.joinpath(filename.strip()).read_bytes(), dtype=np.uint8)
    image = cv.imdecode(image_file, cv.IMREAD_UNCHANGED)
    images_orig.append(image)
  images_orig = np.array(images_orig, dtype=np.float32)
  images_orig /= images_orig.max()
  print("images_orig", images_orig.shape, images_orig.dtype)
  cv.imshow("images_orig", images_orig[0])
  cv.pollKey()
  images = images_orig[:, slice_x, slice_y, :] / light_intensities[:, None, None, :]
  images = np.power(np.clip(images / np.quantile(images, 0.99), 0.0, 1.0), 1 / 2.2)
  images[:, mask, :] = 0
  normal[mask, :] = 0.0
  images = [cv.resize(image, (RENDER_RESOLUTION, RENDER_RESOLUTION), cv.INTER_AREA) for image in images]
  images = np.array(np.clip(np.array(images, dtype=np.float32) * 255, 0, 255), dtype=np.uint8)
  normal = cv.resize(normal, (RENDER_RESOLUTION, RENDER_RESOLUTION), cv.INTER_AREA)
  images = images.transpose(0, 2, 1, 3)[BOUNDARY_IDS, :, ::-1, :]
  normal = normal.transpose(1, 0, 2)[:, ::-1, :]
  normal = normal[:, :, [1, 0, 2]]
  normal[:, :, 1] *= -1
  os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)
  for i in range(len(BOUNDARY_IDS)):
    print("images", images.shape, images.dtype)
    cv.imshow("mask", mask * 1.0)
    cv.imshow("images", images[i])
    cv.imshow("normal", normal * 0.5 + 0.5)
    cv.imwrite(os.path.join(output_dir, "frames", f"{i:06d}.png"), images[i])
    cv.pollKey()
  print("normal", normal.shape, normal.dtype)
  normal_mask = minimum_filter(normal[:, :, 2], size=3, mode="constant") > 1e-3
  normal[~normal_mask, :] = 0.0
  with lzma.LZMAFile(os.path.join(output_dir, "normal.xz"), "w") as f:
    f.write(normal.astype(np.float32).transpose(2, 0, 1).tobytes())
  cv.pollKey()

def main():
  input_file = zipfile.ZipFile(sys.argv[1])
  zip_path = zipfile.Path(input_file).joinpath(sys.argv[2])
  input_dirs = sorted(map(lambda x: x.name, zip_path.iterdir()))
  for i, input_dir in enumerate(input_dirs):
    print("Processing", input_dir)
    process(zip_path.joinpath(input_dir), os.path.join(sys.argv[3], f"{i:06d}"))

if __name__ == "__main__":
  main()
