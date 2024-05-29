#pragma once
#include <memory>
#include <stddef.h>
#include <stdint.h>
#include <metavision/sdk/driver/camera.h>

namespace EventPS {
  struct Camera;
}

#include "event_ps/src/loader/loader_prophesee.rs.h"

namespace EventPS {
  // Must be the same as `EV_BUFFER_SIZE` in `src/loader/loader_prophesee.rs`
  const size_t EV_BUFFER_SIZE = 8192;
  // Must be the same as `EV_BUFFER_N_JOBS` in `src/loader_prophesee/loader.rs`
  const size_t EV_BUFFER_N_JOBS = 4;

  struct CameraData {
    size_t i_ev = 0;
    uint16_t i_row[EV_BUFFER_SIZE];
    uint16_t i_col[EV_BUFFER_SIZE];
    uint8_t polarity[EV_BUFFER_SIZE];
    uint32_t timestamp[EV_BUFFER_SIZE];
  };

  struct Camera {
    Camera(const std::string &serial, rust::Box<RustCamera> rust_camera);
    Metavision::Camera camera;
    Metavision::CallbackId callback_cd, callback_ext_trigger;
    rust::Box<RustCamera> rust_camera;
    CameraData camera_data[EV_BUFFER_N_JOBS];
  };

  std::shared_ptr<Camera> create_camera(
    rust::Box<RustCamera> rust_camera,
    rust::Str serial,
    const uint16_t begin_row,
    const uint16_t end_row,
    const uint16_t begin_col,
    const uint16_t end_col,
    const int32_t bias_fo,
    const int32_t bias_diff_on,
    const int32_t bias_diff_off,
    const int32_t bias_refr);

  void stop_camera(std::shared_ptr<Camera> camera);
}
