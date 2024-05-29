import os
import sys

def main():
  results = {}
  input_file = sys.stdin if len(sys.argv) < 2 else open(sys.argv[1])
  for line in input_file.readlines():
    line = line.split()
    if len(line) != 6 or line[0] != "BENCHMARK":
      continue
    method = line[1]
    n_pixel = int(line[3])
    ang_err = float(line[5])
    if not method in results:
      results[method] = {
        "n_pixel": 0,
        "ang_err": 0.,
      }
    results[method]["n_pixel"] += n_pixel
    results[method]["ang_err"] += n_pixel * ang_err
  input_file.close()
  for v in results.values():
    v["ang_err"] /= v["n_pixel"]
  print(results)

if __name__ == "__main__":
  main()

