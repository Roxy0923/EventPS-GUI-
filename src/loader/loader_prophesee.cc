#include "event_ps/include/loader_prophesee.h"
#include <string>
#include <vector>
#include <stdio.h>
#include <metavision/hal/facilities/i_ll_biases.h>
#include <metavision/hal/facilities/i_monitoring.h>
#include <metavision/hal/facilities/i_trigger_in.h>
#include <metavision/hal/facilities/i_hw_identification.h>

namespace EventPS {
  Camera::Camera(const std::string &serial, rust::Box<RustCamera> rust_camera) :
      camera(serial.empty() ? Metavision::Camera::from_first_available() : Metavision::Camera::from_serial(serial)),
      rust_camera(std::move(rust_camera)) {
  }

  std::shared_ptr<Camera> create_camera(
      rust::Box<RustCamera> rust_camera,
      rust::Str rust_serial,
      const uint16_t begin_row,
      const uint16_t end_row,
      const uint16_t begin_col,
      const uint16_t end_col,
      const int32_t bias_of,
      const int32_t bias_diff_on,
      const int32_t bias_diff_off,
      const int32_t bias_refr) {
    std::string serial = std::string(rust_serial);
    std::shared_ptr<Camera> camera = std::make_shared<Camera>(serial, std::move(rust_camera));
    Metavision::I_HW_Identification *i_hw_identification =
      camera->camera.get_device().get_facility<Metavision::I_HW_Identification>();
    assert(i_hw_identification->get_sensor_info().name_ == "IMX636");
    Metavision::I_LL_Biases *i_ll_biases = camera->camera.get_device().get_facility<Metavision::I_LL_Biases>();
    i_ll_biases->set("bias_fo", bias_of);
    i_ll_biases->set("bias_diff_on", bias_diff_on);
    i_ll_biases->set("bias_diff_off", bias_diff_off);
    i_ll_biases->set("bias_refr", bias_refr);
    Metavision::I_TriggerIn *i_trigger_in = camera->camera.get_device().get_facility<Metavision::I_TriggerIn>();
    i_trigger_in->enable(Metavision::I_TriggerIn::Channel::Main);
    Metavision::I_Monitoring *i_monitoring = camera->camera.get_device().get_facility<Metavision::I_Monitoring>();
    camera->callback_cd = camera->camera.cd().add_callback([camera](
        const Metavision::EventCD *ev_begin,
        const Metavision::EventCD *ev_end) {
      for (const Metavision::EventCD *ev = ev_begin; ev != ev_end; ++ev) {
        const uint16_t i_job = ev->x % EV_BUFFER_N_JOBS;
        CameraData &camera_data = camera->camera_data[i_job];
        size_t &i_ev = camera_data.i_ev;
        camera_data.i_row[i_ev] = ev->y;
        camera_data.i_col[i_ev] = ev->x;
        camera_data.polarity[i_ev] = (ev->p > 0);
        camera_data.timestamp[i_ev] = ev->t;
        if (++i_ev == EV_BUFFER_SIZE) {
          rust::Slice<const uint16_t> slice_i_row(camera_data.i_row, EV_BUFFER_SIZE);
          rust::Slice<const uint16_t> slice_i_col(camera_data.i_col, EV_BUFFER_SIZE);
          rust::Slice<const uint8_t> slice_polarity(camera_data.polarity, EV_BUFFER_SIZE);
          rust::Slice<const uint32_t> slice_timestamp(camera_data.timestamp, EV_BUFFER_SIZE);
          camera->rust_camera = cd_callback(
            std::move(camera->rust_camera),
            i_job,
            slice_i_row,
            slice_i_col,
            slice_polarity,
            slice_timestamp);
          i_ev = 0;
        }
      }
    });
    camera->callback_ext_trigger = camera->camera.ext_trigger().add_callback([camera, i_monitoring](
        const Metavision::EventExtTrigger *trigger_begin,
        const Metavision::EventExtTrigger *trigger_end) {
      for (const Metavision::EventExtTrigger *trigger = trigger_begin; trigger != trigger_end; ++trigger) {
        camera->rust_camera = ext_trigger_callback(
          std::move(camera->rust_camera),
          trigger->t,
          i_monitoring->get_pixel_dead_time());
      }
    });
    camera->camera.roi().set({begin_col, begin_row, end_col - begin_col, end_row - begin_row});
    camera->camera.start();
    return camera;
  }

  void stop_camera(std::shared_ptr<Camera> camera) {
    camera->camera.stop();
    camera->camera.cd().remove_callback(camera->callback_cd);
    camera->camera.ext_trigger().remove_callback(camera->callback_ext_trigger);
  }
}
