import os
import sys
import math
import torch
import cv2 as cv
cv.setNumThreads(0)
# os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # 注释掉以启用GUI
import numpy as np

def map_normal(normal):
  return np.stack([normal[2], -normal[0], normal[1]], axis=0)

def add_data_eval(buffer, light_dir, normal_gt, mask):
  # print("buffer", buffer.shape, buffer.dtype)
  if mask:
    mask = torch.tensor(cv.imread(mask)[:, :, 0] > 128, dtype=torch.bool, device=DEVICE)
    # print("mask", mask.shape, mask.dtype)
  else:
    mask = torch.ones(buffer.shape[2:], dtype=torch.bool, device=DEVICE)
  if normal_gt is not None:
    # print("normal_gt", normal_gt.shape, normal_gt.dtype)
    normal_show = np.clip(0.5 + 0.5 * map_normal(normal_gt).transpose(1, 2, 0), 0., 1.)
    cv.imshow("normal_gt", (normal_show * 255.).astype(np.uint8))
    normal_gt = torch.tensor(normal_gt[None, ...], dtype=torch.float32, device=DEVICE)
    mask = (normal_gt[0, 2, :, :] > 1e-3)
  n_pixels = mask.sum().item()
  for i in range(4):
    buffer_show = np.clip(0.5 + 0.5 * buffer[i * 4, ...].transpose(1, 2, 0), 0., 1.)
    cv.imshow("buffer_"+str(i), (buffer_show * 255.).astype(np.uint8))
  # cv.waitKey(0)
  # n_bin = 16
  # buffer = buffer[:n_bin, ...]
  n_bin = buffer.shape[0]
  n_col = buffer.shape[2]
  n_row = buffer.shape[3]
  event = torch.tensor(buffer, dtype=torch.float32, device=DEVICE)
  down_sample = 1
  if max(n_col, n_row) > 640 and True:
    down_sample = 2
  assert(n_col % down_sample == 0)
  n_col //= down_sample
  assert(n_row % down_sample == 0)
  n_row //= down_sample
  mask = mask.reshape(n_col, down_sample, n_row, down_sample).all(3).all(1)
  event = event.reshape(n_bin, 3, n_col, down_sample, n_row, down_sample).mean(dim=(3, 5))
  if normal_gt is not None:
    normal_gt = normal_gt.reshape(normal_gt.shape[0], 3, n_col, down_sample, n_row, down_sample).mean(dim=(3, 5))
  event[:, :, ~mask] = 0
  # print("light_dir", light_dir[0], light_dir.shape, light_dir.dtype)
  light_dir = np.tile(light_dir[:n_bin, :, None, None], (1, 1, n_col, n_row))
  # print("light_dir", light_dir.shape, light_dir.dtype)
  light = torch.tensor(0.01 * light_dir, dtype=torch.float32, device=DEVICE)
  with torch.no_grad():
    normal = NET(event, light)
  normal[:, :, ~mask] = 0
  normal_show = map_normal(normal[0, ...].cpu().numpy()).transpose(1, 2, 0)
  normal_show = np.clip(0.5 + 0.5 * normal_show, 0., 1.)
  cv.imshow("normal_pred_ps_fcn", (normal_show * 255.).astype(np.uint8))
  cv.waitKey(1)  # 改为waitKey(1)以正确显示GUI
  if normal_gt is not None:
    normal = normal[0, :, mask]
    normal_gt = normal_gt[0, :, mask]
    ang_err_mean = torch.arccos(torch.sum(normal_gt * normal, dim=0)).mean().item() / math.pi * 180
    return n_pixels, ang_err_mean
    # print("BENCHMARK Event-PS-FCN n_pixels", n_pixels, "ang_err_mean", ang_err_mean)

print("__package__", __package__)
print("__name__", __name__)
if __name__ == "__ev_ps_fcn_main__":
  SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
  print("SCRIPT_DIR", SCRIPT_DIR)
  path_last = sys.path
  sys.path.insert(0, SCRIPT_DIR)
  from ev_ps_fcn import EV_PS_FCN
  sys.path = path_last
  DEVICE = torch.device("cuda:0")
  NET = EV_PS_FCN().to(DEVICE).eval()
  NET.load_state_dict(torch.load("data/models/ev_ps_fcn_020000.bin")["model_state_dict"])
  print("Loaded model: ev_ps_fcn_020000.bin", flush=True)
