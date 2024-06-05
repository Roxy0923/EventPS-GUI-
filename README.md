# EventPS: Real-Time Photometric Stereo Using an Event Camera
[\[Project Page\]](https://www.ybh1998.space/eventps-real-time-photometric-stereo-using-an-event-camera/)
[\[Codeberg Repo\]](https://codeberg.org/ybh1998/EventPS/)

Official implementation for EventPS. This repo is still under construction. The code and data is ready, but needs
better document.

### To clone this project with data
This repo uses [Git LFS](https://git-lfs.com/) to host data. To clone the code with data:
```
git lfs install
git clone https://codeberg.org/ybh1998/EventPS.git
```
Or, to clone the code only:
```
GIT_LFS_SKIP_SMUDGE=1 git clone https://codeberg.org/ybh1998/EventPS.git
```

### To build this project from source code

TODO

### To reproduce the device

The device 3D printing files are at `device/stl/`. The Arduino controller programs are at `device/arduino/`.

### The code is tested on the following platforms:

| OS | Device | Driver |
|----|--------|--------|
| Debian Bullseye Linux 6.1.0-0.deb11.7-amd64 | CPU: Intel Core i7-8550U     | PoCL v1.6                      |
| Debian Bullseye Linux 6.1.0-0.deb11.7-amd64 | GPU: NVIDIA GeForce RTX 3090 | NVIDIA Proprietary v470.161.03 |

Copyright (c) 2023-2024 Bohan Yu. All rights reserved.

EventPS is free software licensed under GNU Affero General Public License version 3 or latter.
