import os
import sys
import torch
import cv2 as cv
import numpy as np
import torch.nn as nn
from torch.nn.init import kaiming_normal_

NOISE_STD = 0.1

def map_normal(normal):
  return np.stack([normal[2], -normal[0], normal[1]], axis=0)

def add_data_train(buffer, light_dir, normal_gt):
  global I_ITER
  # print("buffer", buffer.shape, buffer.dtype, buffer.min(), buffer.max())
  # print("normal_gt", normal_gt.shape, normal_gt.dtype)
  buffer += np.random.normal(0.0, NOISE_STD, buffer.shape)
  normal_gt_show = np.clip(0.5 + 0.5 * map_normal(normal_gt).transpose(1, 2, 0), 0., 1.)
  cv.imshow("normal_gt", (normal_gt_show * 255.).astype(np.uint8))
  for i in range(4):
    buffer_show = np.clip(0.5 + 0.5 * buffer[i * 4, ...].transpose(1, 2, 0), 0., 1.)
    cv.imshow("buffer_"+str(i * 4), (buffer_show * 255.).astype(np.uint8))
  # n_bin = 16
  # buffer = buffer[:n_bin, ...]
  n_bin = buffer.shape[0]
  n_col = buffer.shape[2]
  n_row = buffer.shape[3]
  # print("light_dir", light_dir[0], light_dir.shape, light_dir.dtype)
  light_dir = np.tile(light_dir[:n_bin, :, None, None], (1, 1, n_col, n_row))
  # print("light_dir", light_dir.shape, light_dir.dtype)
  event = torch.tensor(buffer, dtype=torch.float32, device=DEVICE)
  light = torch.tensor(0.01 * light_dir, dtype=torch.float32, device=DEVICE)
  normal_gt = torch.tensor(normal_gt.reshape(1, 3, n_col, n_row), dtype=torch.float32, device=DEVICE)
  mask = (normal_gt[:, 2, :, :] > 1e-3)
  OPTIMIZER.zero_grad()
  normal = NET(event, light)
  loss = 1 - torch.sum(normal_gt * normal, dim=1)
  # loss = torch.sum((normal_gt - normal).abs(), dim=1)
  loss_mean = loss[mask].mean()
  print("iter", I_ITER, "loss", loss.shape, "loss_mean", loss_mean.item())
  loss_mean.backward()
  OPTIMIZER.step()
  SCHEDULER.step()
  # print("normal", normal.shape, normal.min(), normal.max())
  normal = normal.detach()
  normal[~torch.stack([mask] * 3, dim=1)] = 0.0
  normal_show = np.clip(0.5 + 0.5 * map_normal(normal[0, ...].cpu().numpy()).transpose(1, 2, 0), 0., 1.)
  cv.imshow("normal", (normal_show * 255.).astype(np.uint8))
  cv.pollKey()
  if I_ITER % 3000 == 0:
    state_dict = {
      "iter": I_ITER,
      "model_state_dict": NET.state_dict(),
      "optimizer_state_dict": OPTIMIZER.state_dict(),
    }
    torch.save(state_dict, f"data/models/ev_ps_fcn_{I_ITER:06d}.bin")
  I_ITER += 1

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
  NET = EV_PS_FCN().to(DEVICE).train()
  checkpoint = torch.load(os.path.join(SCRIPT_DIR, "ev_ps_fcn", "PS-FCN", "data", "models", "PS-FCN_B_S_32.pth.tar"))
  # print("checkpoint", dir(checkpoint))
  del checkpoint["state_dict"]["extractor.conv1.0.weight"]
  NET.load_state_dict(checkpoint["state_dict"], strict=False)
  # print(list(NET.extractor.named_parameters()))
  # exit()
  for m in list(NET.conv_time.modules()) + list(NET.extractor.conv1.modules()) + list(NET.extractor.conv2.modules()):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        kaiming_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
  extractor_params = []
  for (name, param) in NET.extractor.named_parameters():
    if not name.startswith("conv1") and not name.startswith("conv2"):
      extractor_params.append(param)
  params = [
    {"params": NET.conv_time.parameters(), "lr": 1e-4},
    {"params": NET.extractor.conv1.parameters(), "lr": 1e-4},
    {"params": NET.extractor.conv2.parameters(), "lr": 1e-4},
    {"params": extractor_params, "lr": 1e-5},
    {"params": NET.regressor.parameters(), "lr": 1e-6},
  ]
  OPTIMIZER = torch.optim.Adam(params, amsgrad=True)
  SCHEDULER = torch.optim.lr_scheduler.MultiStepLR(OPTIMIZER, milestones=[10000], gamma=0.1)
  I_ITER = 0
