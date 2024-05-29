import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_

PS_FCN = importlib.import_module(".PS-FCN.models.PS_FCN_run", __package__)
FeatExtractor = PS_FCN.FeatExtractor
Regressor = PS_FCN.Regressor

class EV_PS_FCN(nn.Module):
  def __init__(self, fuse_type='max', batchNorm=False, c_in=6, other={}):
    super().__init__()
    self.conv_time = nn.Sequential(
      nn.Conv1d(c_in, 64, kernel_size=3, padding=1),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv1d(64, 64, kernel_size=3, padding=1),
      nn.LeakyReLU(0.1, inplace=True))
    self.extractor = FeatExtractor(batchNorm, 64, other)
    self.regressor = Regressor(batchNorm, other)
    self.c_in      = c_in
    self.fuse_type = fuse_type
    self.other = other

    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        kaiming_normal_(m.weight.data)
        if m.bias is not None:
          m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def forward(self, event, light):
    net_in = torch.cat([event, light], 1)
    n_bin, _, n_row, n_col = net_in.shape
    # print("net_in", net_in.shape)
    net_in = net_in.permute(2, 3, 1, 0).reshape(-1, 6, n_bin)
    # print("net_in", net_in.shape)
    net_in = self.conv_time(net_in)
    # print("net_in", net_in.shape)
    net_in = net_in.reshape(n_row, n_col, 64, n_bin).permute(3, 2, 0, 1)
    # print("net_in", net_in.shape)
    feats = []
    for net_in in torch.split(net_in, 1, 0):
      feat, shape = self.extractor(net_in)
      # print("feat", feat.shape, "shape", shape)
      feats.append(feat)
    if self.fuse_type == 'mean':
      feat_fused = torch.stack(feats, 1).mean(1)
    elif self.fuse_type == 'max':
      feat_fused, _ = torch.stack(feats, 1).max(1)
    normal = self.regressor(feat_fused, shape)
    return normal
