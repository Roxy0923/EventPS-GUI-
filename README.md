# EventPS: Real-Time Photometric Stereo Using an Event Camera
[\[Project Page\]](https://www.ybh1998.space/eventps-real-time-photometric-stereo-using-an-event-camera/)
[\[Codeberg Repo\]](https://codeberg.org/ybh1998/EventPS/)

Official implementation for EventPS. This repo is still under construction. The code and data are ready, but we need
better document.

### To clone this project with data
This repo uses [Git LFS](https://git-lfs.com/) to host data. To clone the code with data:
```
git lfs install
git clone --recursive https://codeberg.org/ybh1998/EventPS.git
```
Or, to clone the code only:
```
GIT_LFS_SKIP_SMUDGE=1 git clone --recursive https://codeberg.org/ybh1998/EventPS.git
```

### To build this project from source code
This project is written in Rust and Python. For a complete build:
```
cargo install --path . --features display_cv,display_gl,loader_render,loader_prophesee
```
The optional features are:

#### display_cv
Display results on the Rust side with [OpenCV](https://opencv.org/). When enabled, `show_ls_ps = cv` is available.

#### display_gl
Display results on the Rust side with [OpenGL](https://www.opengl.org/). When enabled, `show_ls_ps = gl` is available.
The default OpenGL rendering device must be the same as the OpenCL device. This option prevents data copying and has a
better refresh rate.

#### loader_render
Rendering events for the Blobby, Sculpture, and DiLiGenT datasets. This option requires a running
[LibreDR](https://codeberg.org/ybh1998/LibreDR/) server and worker.

#### loader_prophesee
Capturing events from a [PROPHESEE](https://www.prophesee.ai/) camera in real-time. This option requires
[OpenEB](https://github.com/prophesee-ai/openeb).

### To benchmark on the DiLiGenT dataset
During building, `display_cv` and `loader_render` features are required for this benchmark. The pre-trained models are at `data/models/*.bin`. Download the `DiLiGenT.zip` from [DiLiGenT](https://sites.google.com/site/photometricstereodata/single) to `data/DiLiGenT.zip`. Run the following scripts:
```
bash ./scripts/diligent_convert.sh
bash ./scripts/diligent_eval.sh
```
The results will be saved to `data/diligent/*/result.txt`.

### To train deep-learning models
Rendered data is required to train the models. The processed 3D object files are at `data/{blobs,sculpture}_processed/`. To render the training and evaluation data, make sure to have a working [LibreDR](https://codeberg.org/ybh1998/LibreDR/) server and worker, and run the following command:
```
 bash ./scripts/render.sh
```
To train the EventPS-FCN model, download the pre-trained original PS-FCN model, following the instructions in [PS-FCN](https://github.com/guanyingc/PS-FCN). File `python/ev_ps_fcn/PS-FCN/data/models/PS-FCN_B_S_32.pth.tar`is required. Then run the following command:
```
event_ps_train --ps-fcn-train python/ps_fcn_train.py
```
To train the EventPS-CNN model, run the following command:
```
event_ps_train --cnn-ps-train python/cnn_ps_train.py
```

### To reproduce the device
The device's 3D printing files are at `device/stl/`. The demo device is 3D printed with carbon fiber filament. The
[Arduino](https://www.arduino.cc/) controller programs are at `device/arduino/`.

### The code is tested on the following platforms:

| OS | Device | Driver |
|----|--------|--------|
| Debian Bullseye Linux 6.1.0-0.deb11.7-amd64 | CPU: Intel Core i7-8550U     | PoCL v1.6                      |
| Debian Bullseye Linux 6.1.0-0.deb11.7-amd64 | GPU: NVIDIA GeForce RTX 3090 | NVIDIA Proprietary v470.161.03 |

Copyright (c) 2023-2024 Bohan Yu. All rights reserved.

EventPS is free software licensed under the GNU Affero General Public License, version 3 or any later version.
