import os
import sys
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

BATCH_SIZE = 4096

def map_normal(normal):
  return np.stack([normal[2], -normal[0], normal[1]], axis=0)

def add_data_train(buffer, normal_gt):
  global I_ITER
  # print("buffer", buffer.shape, buffer.dtype, buffer.min(), buffer.max())
  # print("normal_gt", normal_gt.shape, normal_gt.dtype)
  buffer_show = buffer[:, :, :, ::8, ::8].transpose(3, 1, 4, 2, 0)
  buffer_show = buffer_show.reshape(buffer_show.shape[0] * buffer_show.shape[1],
                                    buffer_show.shape[2] * buffer_show.shape[3],
                                    buffer_show.shape[4])
  buffer_show = np.clip(0.5 + 2 * buffer_show, 0., 1.)
  cv.imshow("buffer_show", buffer_show)
  normal_gt_show = np.clip(0.5 + 0.5 * map_normal(normal_gt).transpose(1, 2, 0), 0., 1.)
  cv.imshow("normal_gt", (normal_gt_show * 255.).astype(np.uint8))
  buffer = torch.tensor(buffer, dtype=torch.float32, device=DEVICE)
  normal_gt = torch.tensor(normal_gt, dtype=torch.float32, device=DEVICE)
  normal_pred = torch.zeros_like(normal_gt)
  mask = (normal_gt[2, :, :] > 1e-3)
  normal_gt = normal_gt[:, mask].permute(1, 0)
  normal_pred_gather = torch.zeros_like(normal_gt)
  buffer = buffer[:, :, :, mask].permute(3, 0, 1, 2)
  dataset = TensorDataset(buffer, normal_gt)
  dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
  loss_sum = 0
  item_sum = 0
  for i_batch, (buffer, normal_gt) in enumerate(dataloader):
    OPTIMIZER.zero_grad()
    normal = NET(buffer)
    loss = 1 - torch.sum(normal_gt * normal, dim=1)
    loss.mean().backward()
    OPTIMIZER.step()
    SCHEDULER.step()
    with torch.no_grad():
      loss_sum += loss.sum().item()
      item_sum += buffer.shape[0]
      slice_batch = slice(i_batch * BATCH_SIZE, (i_batch + 1) * BATCH_SIZE)
      normal_pred_gather[slice_batch, :] = normal
  # print("normal_pred_gather", normal_pred_gather.shape)
  normal_pred[:, mask] = normal_pred_gather.T
  normal_pred_show = np.clip(0.5 + 0.5 * map_normal(normal_pred.cpu().numpy()).transpose(1, 2, 0), 0., 1.)
  cv.imshow("normal_pred", (normal_pred_show * 255.).astype(np.uint8))
  cv.pollKey()
  print("iter", I_ITER, "item_sum", item_sum, "loss_mean", loss_sum / (item_sum + 1e-6))
  if I_ITER % 300 == 0:
    state_dict = {
      "iter": I_ITER,
      "model_state_dict": NET.state_dict(),
      "optimizer_state_dict": OPTIMIZER.state_dict(),
    }
    torch.save(state_dict, f"data/models/ev_cnn_ps_{I_ITER:06d}.bin")
  I_ITER += 1

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
  NET = EV_CNN_PS().to(DEVICE).train()
  OPTIMIZER = torch.optim.Adam(NET.parameters(), lr=1e-5, amsgrad=True)
  SCHEDULER = torch.optim.lr_scheduler.MultiStepLR(OPTIMIZER, milestones=[1000], gamma=0.1)
  I_ITER = 0

