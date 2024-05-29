import torch
import torch.nn as nn
import torch.nn.functional as F

class EV_CNN_PS(nn.Module):
  def __init__(self):
    super().__init__()
    self.activation = nn.ReLU()
    self.dropout = nn.Dropout(0.2)
    self.pool = nn.AvgPool2d(2)
    self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(16, momentum=1e-3)
    self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
    self.conv4 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
    self.conv5 = nn.Conv2d(48, 16, kernel_size=3, padding=1)
    self.conv6 = nn.Conv2d(64, 16, kernel_size=3, padding=1)
    self.conv7 = nn.Conv2d(80, 80, kernel_size=3, padding=1)
    self.linear1 = nn.Linear(80 * 16 * 16, 128)
    self.linear2 = nn.Linear(128, 3)

  def forward(self, x0, debug=False):
    x1 = self.bn1(self.conv1(x0))
    if debug:
      print("x1", x1.shape)
    x2 = self.dropout(self.conv2(self.activation(x1)))
    if debug:
      print("x2", x2.shape)
    x3 = self.dropout(self.conv3(self.activation(torch.concatenate([x1, x2], dim=1))))
    if debug:
      print("x3", x3.shape)
    x4 = self.dropout(self.conv4(self.activation(torch.concatenate([x1, x2, x3], dim=1))))
    if debug:
      print("x4", x4.shape)
    x1 = self.activation(self.pool(x4))
    if debug:
      print("x1", x1.shape)
    x2 = self.dropout(self.conv5(self.activation(x1)))
    if debug:
      print("x2", x2.shape)
    x3 = self.dropout(self.conv6(self.activation(torch.concatenate([x1, x2], dim=1))))
    if debug:
      print("x3", x3.shape)
    x4 = self.dropout(self.conv7(self.activation(torch.concatenate([x1, x2, x3], dim=1))))
    if debug:
      print("x4", x4.shape)
    x = self.linear1(x4.view(x4.shape[0], 80 * 16 * 16))
    x = F.normalize(self.linear2(self.activation(x)), dim=1)
    return x
