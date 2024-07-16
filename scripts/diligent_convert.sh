#!/bin/bash

set -euxo pipefail

rm -rf data/diligent/
mkdir -p data/diligent/
python3 python/diligent_convert.py data/DiLiGenT.zip DiLiGenT/pmsData/ data/diligent/
for data_dir in data/diligent/* ; do
  event_ps_eval $data_dir/render.ini
  ln -sf event_internal.xz $data_dir/event.xz
done
