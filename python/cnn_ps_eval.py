import os
import sys
import math
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

BATCH_SIZE = 4096

def map_normal(normal):
  return np.stack([normal[2], -normal[0], normal[1]], axis=0)

def add_data_eval(buffer, normal_gt, mask):
  # print("buffer", buffer.shape, buffer.dtype, buffer.min(), buffer.max())
  n_col = buffer.shape[3]
  n_row = buffer.shape[4]
  down_sample = 1
  if max(n_col, n_row) > 640 and True:
    down_sample = 2
  assert(n_col % down_sample == 0)
  n_col //= down_sample
  assert(n_row % down_sample == 0)
  n_row //= down_sample
  if mask:
    mask = torch.tensor(cv.imread(mask)[:, :, 0] > 128, dtype=torch.bool, device=DEVICE)
    # print("mask", mask.shape, mask.dtype)
  else:
    mask = torch.ones(buffer.shape[3:], dtype=torch.bool, device=DEVICE)
  if normal_gt is not None:
    # print("normal_gt", normal_gt.shape, normal_gt.dtype)
    normal_show = np.clip(0.5 + 0.5 * map_normal(normal_gt).transpose(1, 2, 0), 0., 1.)
    cv.imshow("normal_gt", (normal_show * 255.).astype(np.uint8))
    mask = (normal_gt[2, :, :] > 1e-3)
    normal_gt = torch.tensor(normal_gt, dtype=torch.float32, device=DEVICE)
    normal_gt = normal_gt.reshape(3, n_col, down_sample, n_row, down_sample).mean(dim=(2, 4))
  mask = mask.reshape(n_col, down_sample, n_row, down_sample).all(3).all(1)
  n_pixels = mask.sum().item()
  # buffer_show = buffer.reshape(*buffer.shape[:3], buffer.shape[3] // 32, 32, buffer.shape[4] // 32, 32)
  # buffer_show = buffer_show.mean(axis=(4, 6)).transpose(3, 1, 4, 2, 0)
  # buffer_show = buffer_show.reshape(buffer_show.shape[0] * buffer_show.shape[1],
  #                                   buffer_show.shape[2] * buffer_show.shape[3],
  #                                   buffer_show.shape[4])
  # buffer_show = buffer_show.reshape(buffer_show.shape[0] // 2, 2, buffer_show.shape[1] // 2, 2, buffer_show.shape[2])
  # buffer_show = buffer_show.sum(axis=(1, 3))
  # buffer_show = np.clip(0.5 + 10. * buffer_show, 0., 1.) ** (1. / 2.2)
  # cv.imshow("buffer_show", buffer_show)
  buffer = torch.tensor(buffer, dtype=torch.float32, device=DEVICE)
  buffer = buffer.reshape(*buffer.shape[:3], n_col, down_sample, n_row, down_sample).mean(dim=(4, 6))
  buffer = buffer[:, :, :, mask].permute(3, 0, 1, 2)
  dataset = TensorDataset(buffer)
  dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
  normal_pred = torch.zeros([3, mask.shape[0], mask.shape[1]], dtype=torch.float32, device=DEVICE)
  normal_pred_gather = torch.zeros([n_pixels, 3], dtype=torch.float32, device=DEVICE)
  for i_batch, (buffer,) in enumerate(dataloader):
    with torch.no_grad():
      # print("buffer", buffer.shape)
      normal = NET(buffer)
      slice_batch = slice(i_batch * BATCH_SIZE, (i_batch + 1) * BATCH_SIZE)
      normal_pred_gather[slice_batch, :] = normal
  normal_pred[:, mask] = normal_pred_gather.T
  normal_pred_show = np.clip(0.5 + 0.5 * map_normal(normal_pred.cpu().numpy()).transpose(1, 2, 0), 0., 1.)
  cv.imshow("normal_pred_cnn_ps", (normal_pred_show * 255.).astype(np.uint8))
  cv.pollKey()
  if normal_gt is not None:
    normal_gt = normal_gt[:, mask].T
    ang_err_mean = torch.arccos((normal_pred_gather * normal_gt).sum(dim=1)).mean().item() / math.pi * 180
    return n_pixels, ang_err_mean
    # print("BENCHMARK Event-CNN-PS n_pixels", n_pixels, "ang_err_mean", ang_err_mean)

print("__package__", __package__)
print("__name__", __name__)

if __name__ == "__ev_cnn_ps_main__":
  SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
  print("SCRIPT_DIR", SCRIPT_DIR)
  path_last = sys.path
  sys.path.insert(0, SCRIPT_DIR)
  from ev_cnn_ps import EV_CNN_PS
  sys.path = path_last
  DEVICE = torch.device("cuda:0")
  NET = EV_CNN_PS().to(DEVICE).eval()
  NET.load_state_dict(torch.load("data/models/ev_cnn_ps.bin")["model_state_dict"])

