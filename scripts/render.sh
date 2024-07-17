#!/bin/bash

set -euxo pipefail

cd "$(dirname "$0")/../"

rm -rf data/blobs_training/ data/blobs_eval/ data/sculptures_training/ data/sculptures_eval/
mkdir data/blobs_training/ data/blobs_eval/ data/sculptures_training/ data/sculptures_eval/
python3 python/render_generate_config.py
find data/blobs_training/ -name render.ini | sort | xargs -n1 event_ps_eval
find data/blobs_eval/ -name render.ini | sort | xargs -n1 event_ps_eval
find data/sculptures_training/ -name render.ini | sort | xargs -n1 event_ps_eval
find data/sculptures_eval/ -name render.ini | sort | xargs -n1 event_ps_eval
for data_dir in data/{blobs,sculptures}_{training,eval}/* ; do
  ln -sf event_internal.xz $data_dir/event.xz
done
