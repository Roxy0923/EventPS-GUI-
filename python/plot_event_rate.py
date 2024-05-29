import sys
import lzma
import cv2 as cv
import numpy as np
from event_voxel_builder import EventVoxelBuilder

def main():
  with lzma.LZMAFile(sys.argv[1], "r") as f:
    events = np.frombuffer(f.read(), dtype=np.float32).reshape(-1, 4)
  print(events[0])
  i_row = events[:, 0]
  i_col = events[:, 1]
  timestamp = (events[:, 3] * 1e6)
  timestamp -= timestamp.min()
  polarity = abs(events[:, 2])
  n_row = int(i_row.max() + 1)
  n_col = int(i_col.max() + 1)
  event_voxel_builder = EventVoxelBuilder(n_time=10, n_row=n_row, n_col=n_col, timestamp_per_time=int(1e6))
  voxel = event_voxel_builder.build(timestamp, i_row, i_col, polarity)
  for i, image in enumerate(voxel):
    image = (np.clip(image / (3 * image.mean() + 1e-6), 0., 1.) * 255).astype(np.uint8)
    image = cv.applyColorMap(image, cv.COLORMAP_TURBO)
    cv.imshow(f"rate_{i}", image)
  cv.waitKey(0)
  # breakpoint()

if __name__ == "__main__":
  main()
