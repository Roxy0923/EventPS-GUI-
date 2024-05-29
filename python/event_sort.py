import sys
import lzma
import numpy as np

def main():
  with lzma.LZMAFile(sys.argv[1], "r") as f:
    events = np.frombuffer(f.read(), dtype=np.float32).reshape(-1, 4)
  print("events", events.shape, events[:, 0].max(), events[:, 1].max(), events[:, 2].max(), events[:, 3].max())
  events = events[events[:, -1].argsort(), :]
  # f.write(normal.astype(np.float32).transpose(2, 0, 1).tobytes())
  print("events", events[:10])
  with lzma.LZMAFile(sys.argv[1], "w") as f:
    f.write(events.tobytes())


if __name__ == "__main__":
  main()
