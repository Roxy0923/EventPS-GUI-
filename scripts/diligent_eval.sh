#!/bin/bash

set -euxo pipefail

for data_dir in data/diligent/* ; do
  event_ps_eval $data_dir/eval.ini data/enable_all.ini |
    python3 -u python/benchmark_gather_result.py |
    tee $data_dir/result.txt
done
