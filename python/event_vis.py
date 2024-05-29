import sys
import lzma
import cv2 as cv
import numpy as np
from configparser import ConfigParser
from event_voxel_builder import EventVoxelBuilder

def main():
  config = ConfigParser()
  config.read(sys.argv[1])
  duration = float(config["loader_render"]["duration"])
  n_frames = int(config["loader_render"]["n_frames"])
  width = int(config["loader_event_reader"]["width"])
  height = int(config["loader_event_reader"]["height"])
  load_event = config["loader_event_reader"]["load_event"].split(" ")
  with lzma.LZMAFile(load_event[0], "rb") as f:
    events = np.frombuffer(f.read(), dtype=np.float32).reshape(-1, 4)
  print("events", events.shape, events[:, 0].max(), events[:, 1].max(), events[:, 2].max(), events[:, 3].max())
  events = events[events[:, -1].argsort(), :]
  # f.write(normal.astype(np.float32).transpose(2, 0, 1).tobytes())
  print("events", events[:10])
  i_row = events[:, 0]
  i_col = events[:, 1]
  polarity = events[:, 2]
  timestamp = events[:, 3] * n_frames / duration
  event_voxel_builder = EventVoxelBuilder(n_time=n_frames, n_row=height, n_col=width, timestamp_per_time=1)
  voxel = event_voxel_builder.build(timestamp, i_row, i_col, polarity)
  print("voxel", voxel.shape)
  fourcc = cv.VideoWriter_fourcc(*"mp4v")
  writer = cv.VideoWriter("vis/event_vis.mp4", fourcc, 30, (width,height))
  for event_vis in voxel:
    event_vis = event_vis * 0.2
    event_vis = 1.0 - np.stack([
      event_vis.clip(min=0.0, max=1.0),
      np.abs(event_vis).clip(max=1.0),
      -event_vis.clip(min=-1.0, max=0.0)], axis=-1)
    event_vis = (event_vis * 255).astype(np.uint8)
    # event_vis = event_vis.transpose(1, 0, 2)[::-1, ...]
    cv.imshow("event_vis", event_vis)
    cv.pollKey()
    # cv.waitKey(100)
    writer.write(event_vis)
  writer.release()
  # breakpoint()



if __name__ == "__main__":
  main()
