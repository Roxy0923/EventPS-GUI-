use std::collections::HashMap;
use anyhow::{Result, ensure};
use aho_corasick::AhoCorasick;
use super::{EV_BUFFER_N_JOBS, EV_BUFFER_N_BUCKET, EV_BUFFER_BUCKET_SIZE, OBS_MAP_RESOLUTION};
use self::jit_kernel_ls_ps::jit_kernel_ls_ps;
use self::jit_kernel_ps_fcn::jit_kernel_ps_fcn;
use self::jit_kernel_cnn_ps::jit_kernel_cnn_ps;

mod jit_kernel_ls_ps;
mod jit_kernel_ps_fcn;
mod jit_kernel_cnn_ps;

fn round_to_power(mut val: u32) -> u32 {
  val -= 1;
  val |= val >> 1;
  val |= val >> 2;
  val |= val >> 4;
  val |= val >> 8;
  val |= val >> 16;
  val += 1;
  val
}

pub fn jit_kernel_sources(
    n_row: u16,
    n_col: u16,
    config_ps: &HashMap<String, String>) -> Result<String> {
  let enable_ls_ps = matches!(config_ps["show_ls_ps"].as_str(), "gl" | "cv" | "none");
  let enable_ps_fcn = config_ps["ps_fcn_per_n_bin"].parse::<u32>()? > 0;
  let enable_cnn_ps = config_ps["cnn_ps_per_n_bin"].parse::<u32>()? > 0;
  let mut sources = String::new();
  sources.push_str(r###"
typedef struct __attribute__((__packed__)) {
  const ushort n_ev[EV_BUFFER_N_BUCKET];
  const ushort i_row[EV_BUFFER_BUCKET_SIZE][EV_BUFFER_N_BUCKET];
  const ushort i_col[EV_BUFFER_BUCKET_SIZE][EV_BUFFER_N_BUCKET];
  const uchar polarity[EV_BUFFER_BUCKET_SIZE][EV_BUFFER_N_BUCKET];
  const uint timestamp[EV_BUFFER_BUCKET_SIZE][EV_BUFFER_N_BUCKET];
} EVChunk;

typedef struct __attribute__((__packed__)) {
  uchar polarity_last[N_ROW][N_COL / N_JOBS];
  uint timestamp_last[N_ROW][N_COL / N_JOBS];"###);
  if enable_ls_ps {
    sources.push_str(r###"
  float vis_moving[N_ROW][N_COL / N_JOBS];"###);
    /*
      Only storing the upper matrix
      xx xy xz
         yy yz
            zz
    */
    sources.push_str(r###"
  float ls_ps_moving[6][N_ROW][N_COL / N_JOBS];"###);
  }
  if enable_ps_fcn {
    sources.push_str(r###"
  float null_vector_bin[RECORD_N_BIN][3][N_ROW][N_COL / N_JOBS];"###);
  }
  if enable_cnn_ps {
    sources.push_str(r###"
  uint obs_map_timestamp_last[OBS_MAP_RESOLUTION][OBS_MAP_RESOLUTION][N_ROW][N_COL / N_JOBS];
  float obs_map[3][OBS_MAP_RESOLUTION][OBS_MAP_RESOLUTION][N_ROW][N_COL / N_JOBS];"###);
  }
  sources.push_str(r###"
} Pixel;

void weighted_add(global float *target, const float weight, const float add) {
  *target = *target * weight + add;
}

float3 get_light_dir(
    const ushort i_row,
    const ushort i_col,
    const uint timestamp_begin,
    const uint timestamp_end,
    const uint timestamp) {
  const float time = fmod((float)((int)timestamp - (int)timestamp_begin) / (timestamp_end - timestamp_begin), 1.f);
  float3 light_dir;"###);
  sources.push_str(
  match config_ps["scan_pattern"].as_str() {
    "circle_with_calibration" => r###"
  const float longitude = time * 2.f * M_PI_F;
  const float3 light_position = (float3)(
    CIRCLE_LIGHT_DIAMETER / 2 * sin(longitude),
    CIRCLE_LIGHT_DIAMETER / 2 * cos(longitude),
    CIRCLE_OBJECT_DISTANCE);
  const float3 object_position = (float3)(
    ((float)i_row + 0.5f - (float)N_ROW * 0.5f) * PIXEL_WIDTH,
    ((float)i_col + 0.5f - (float)N_COL * 0.5f) * PIXEL_WIDTH,
    0.f);
  light_dir = (light_position - object_position);
  light_dir *= length(light_position) / dot(light_dir, light_dir);"###,
    "hypotrochoid" => r###"
  const float longitude = time * 2.f * M_PI_F;
  const float ratio = -1.f / 1.5f;
  const float x = HYPOTROCHOID_R_BIG * sin(ratio * longitude) + HYPOTROCHOID_R_SMALL * sin(longitude);
  const float y = HYPOTROCHOID_R_BIG * cos(ratio * longitude) + HYPOTROCHOID_R_SMALL * cos(longitude);
  const float z = sqrt(1.f - pown(x, 2) - pown(y, 2));
  light_dir = (float3)(x, y, z);"###,
    "diligent" => r###"
  if (time < 7.f / 36.f) {
    const float ratio = time / (7.f / 36.f);
    light_dir = (1.f - ratio) * (float3)( 0.7794f, -0.4861f, 1.f) + ratio * (float3)( 0.7317f,  0.5074f, 1.f);
  } else if (time < 18.f / 36.f) {
    const float ratio = (time - 7.f / 36.f) / (11.f / 36.f);
    light_dir = (1.f - ratio) * (float3)( 0.7317f,  0.5074f, 1.f) + ratio * (float3)(-0.7421f,  0.5228f, 1.f);
  } else if (time < 25.f / 36.f) {
    const float ratio = (time - 18.f / 36.f) / (7.f / 36.f);
    light_dir = (1.f - ratio) * (float3)(-0.7421f,  0.5228f, 1.f) + ratio * (float3)(-0.8072f, -0.4773f, 1.f);
  } else {
    const float ratio = (time - 25.f / 36.f) / (11.f / 36.f);
    light_dir = (1.f - ratio) * (float3)(-0.8072f, -0.4773f, 1.f) + ratio * (float3)( 0.7794f, -0.4861f, 1.f);
  }
  light_dir = normalize(light_dir);"###,
    _ => todo!(),
  });
  sources.push_str(r###"
  return light_dir;
}

float3 get_null_vector(
    const ushort i_row,
    const ushort i_col,
    const uchar polarity,
    const uint timestamp,
    const uint timestamp_last,
    const uint timestamp_begin,
    const uint timestamp_end) {
  const float3 last_light_dir = get_light_dir(i_row, i_col, timestamp_begin, timestamp_end, timestamp_last);
  const float3 curr_light_dir = get_light_dir(i_row, i_col, timestamp_begin, timestamp_end, timestamp);
  if (polarity == 1) {
    return curr_light_dir - EXP_EVENT_THRESHOLD * last_light_dir;
  }
  return EXP_EVENT_THRESHOLD * curr_light_dir - last_light_dir;
}

kernel void kernel_process_chunk(
    global const EVChunk *ev_chunk,
    global Pixel *pixel,
    const uint timestamp_begin,
    const uint timestamp_end) {
  const ushort i_bucket = get_global_id(0);
  for (ushort i_ev = 0; i_ev < ev_chunk->n_ev[i_bucket]; ++i_ev) {
    const ushort i_row = ev_chunk->i_row[i_ev][i_bucket];
    const ushort i_col_orig = ev_chunk->i_col[i_ev][i_bucket];
    const ushort i_col = i_col_orig / N_JOBS;
    const uint timestamp = ev_chunk->timestamp[i_ev][i_bucket];
    const uint timestamp_last = pixel->timestamp_last[i_row][i_col];"###);
  if enable_ps_fcn {
    sources.push_str(r###"
    const uint i_bin = timestamp / TIME_PER_BIN % RECORD_N_BIN;
    const uint i_bin_last = timestamp_last / TIME_PER_BIN % RECORD_N_BIN;
    for (uint i_bin_erase = (i_bin_last + 1) % RECORD_N_BIN;
         i_bin_erase != (i_bin + 1) % RECORD_N_BIN;
         i_bin_erase = (i_bin_erase + 1) % RECORD_N_BIN) {
      pixel->null_vector_bin[i_bin_erase][0][i_row][i_col] = 0.f;
      pixel->null_vector_bin[i_bin_erase][1][i_row][i_col] = 0.f;
      pixel->null_vector_bin[i_bin_erase][2][i_row][i_col] = 0.f;
    }"###);
  }
  sources.push_str(r###"
    pixel->timestamp_last[i_row][i_col] = timestamp;
    const uchar polarity = ev_chunk->polarity[i_ev][i_bucket];
    const uchar polarity_last = pixel->polarity_last[i_row][i_col];
    pixel->polarity_last[i_row][i_col] = polarity + 1;"###);

  if enable_ls_ps {
    sources.push_str(r###"
    const float weight_vis = exp2(-(float)(timestamp - timestamp_last) / VIS_HALF_LIFE);
    weighted_add(&pixel->vis_moving[i_row][i_col], weight_vis, polarity * 2.f - 1.f);"###);
  }
  sources.push_str(r###"
    if (timestamp_end - timestamp_begin < EVENT_REFRACTORY_THRESHOLD_MIN ||
        timestamp - timestamp_last > EVENT_REFRACTORY_THRESHOLD_MAX) {
      continue;
    }
    const float3 null_vector = get_null_vector(i_row,
                                               i_col_orig,
                                               polarity,
                                               timestamp,
                                               timestamp_last + EVENT_REFRACTORY_U32,
                                               timestamp_begin,
                                               timestamp_end);"###);
  if enable_ps_fcn {
    sources.push_str(r###"
    if (polarity_last == polarity + 1) {
      pixel->null_vector_bin[i_bin][0][i_row][i_col] += null_vector.x;
      pixel->null_vector_bin[i_bin][1][i_row][i_col] += null_vector.y;
      pixel->null_vector_bin[i_bin][2][i_row][i_col] += null_vector.z;
    }"###);
  }
  if enable_cnn_ps {
    sources.push_str(r###"
    if (polarity_last == polarity + 1) {
      const float3 light_dir = normalize(get_light_dir(i_row, i_col_orig, timestamp_begin, timestamp_end, timestamp));
      const uint i_row_obs = floor(clamp((light_dir.x + 1.f) / 2.f, 1e-6f, 1.f - 1e-6f) * OBS_MAP_RESOLUTION);
      const uint i_col_obs = floor(clamp((light_dir.y + 1.f) / 2.f, 1e-6f, 1.f - 1e-6f) * OBS_MAP_RESOLUTION);
      const uint obs_map_timestamp_last = pixel->obs_map_timestamp_last[i_row_obs][i_col_obs][i_row][i_col];
      pixel->obs_map_timestamp_last[i_row_obs][i_col_obs][i_row][i_col] = timestamp;
      const float weight = exp2(-(float)(timestamp - obs_map_timestamp_last) / CNN_PS_HALF_LIFE);
      weighted_add(&pixel->obs_map[0][i_row_obs][i_col_obs][i_row][i_col], weight, null_vector.x);
      weighted_add(&pixel->obs_map[1][i_row_obs][i_col_obs][i_row][i_col], weight, null_vector.y);
      weighted_add(&pixel->obs_map[2][i_row_obs][i_col_obs][i_row][i_col], weight, null_vector.z);
    }"###);
  }
  if enable_ls_ps {
    sources.push_str(r###"
    if (timestamp - timestamp_last >= EVENT_REFRACTORY_THRESHOLD_MIN) {
      const float weight_ls_ps = exp2(-(float)(timestamp - timestamp_last) / LS_PS_HALF_LIFE);
      weighted_add(&pixel->ls_ps_moving[0][i_row][i_col], weight_ls_ps, null_vector.x * null_vector.x);
      weighted_add(&pixel->ls_ps_moving[1][i_row][i_col], weight_ls_ps, null_vector.x * null_vector.y);
      weighted_add(&pixel->ls_ps_moving[2][i_row][i_col], weight_ls_ps, null_vector.x * null_vector.z);
      weighted_add(&pixel->ls_ps_moving[3][i_row][i_col], weight_ls_ps, null_vector.y * null_vector.y);
      weighted_add(&pixel->ls_ps_moving[4][i_row][i_col], weight_ls_ps, null_vector.y * null_vector.z);
      weighted_add(&pixel->ls_ps_moving[5][i_row][i_col], weight_ls_ps, null_vector.z * null_vector.z);
    }"###);
  }
  sources.push_str(r###"
  }
}
"###);
  if enable_ls_ps {
    sources.push_str(&jit_kernel_ls_ps(config_ps));
  }
  if enable_ps_fcn {
    sources.push_str(&jit_kernel_ps_fcn());
  }
  if enable_cnn_ps {
    sources.push_str(&jit_kernel_cnn_ps());
  }
  let record_time = config_ps["record_time"].parse::<u32>()?;
  ensure!(record_time == round_to_power(record_time), "PSCL::jit_kernel_sources: `record_time` must be power of 2");
  let record_n_bin = config_ps["record_n_bin"].parse::<u32>()?;
  ensure!(record_time % record_n_bin == 0, "PSCL::jit_kernel_sources: record_time % record_n_bin == 0");
  let time_per_bin = record_time / record_n_bin;
  let (patterns, replace_with): (Vec<_>, Vec<_>) = [
    ("N_ROW", format!("{n_row:?}u")),
    ("N_COL", format!("{n_col:?}u")),
    ("N_JOBS", format!("{EV_BUFFER_N_JOBS:?}u")),
    ("EV_BUFFER_N_BUCKET", format!("{EV_BUFFER_N_BUCKET:?}u")),
    ("EV_BUFFER_BUCKET_SIZE", format!("{EV_BUFFER_BUCKET_SIZE:?}u")),
    ("CIRCLE_LIGHT_DIAMETER", format!("{:?}f", config_ps["circle_light_diameter"].parse::<f32>()?)),
    ("CIRCLE_OBJECT_DISTANCE", format!("{:?}f", config_ps["circle_object_distance"].parse::<f32>()?)),
    ("HYPOTROCHOID_R_BIG", format!("{:?}f", config_ps["hypotrochoid_r_big"].parse::<f32>()?)),
    ("HYPOTROCHOID_R_SMALL", format!("{:?}f", config_ps["hypotrochoid_r_small"].parse::<f32>()?)),
    ("PIXEL_WIDTH", format!("{:?}f", config_ps["circle_view_width"].parse::<f32>()? / n_col as f32)),
    ("EXP_EVENT_THRESHOLD", format!("{:?}f", config_ps["event_threshold"].parse::<f32>()?.exp())),
    ("EVENT_REFRACTORY_THRESHOLD_MIN", format!("{:?}u", config_ps["event_refractory_threshold_min"].parse::<u32>()?)),
    ("EVENT_REFRACTORY_THRESHOLD_MAX", format!("{:?}u", config_ps["event_refractory_threshold_max"].parse::<u32>()?)),
    ("EVENT_REFRACTORY_U32", format!("{:?}u", config_ps["event_refractory"].parse::<u32>()?)),
    ("RECORD_TIME", format!("{record_time:?}u")),
    ("RECORD_N_BIN", format!("{record_n_bin:?}u")),
    ("VIS_HALF_LIFE", format!("{:?}u", config_ps["vis_half_life"].parse::<u32>()?)),
    ("LS_PS_HALF_LIFE", format!("{:?}u", config_ps["ls_ps_half_life"].parse::<u32>()?)),
    ("CNN_PS_HALF_LIFE", format!("{:?}u", config_ps["cnn_ps_half_life"].parse::<u32>()?)),
    ("TIME_PER_BIN", format!("{time_per_bin:?}u")),
    ("OBS_MAP_RESOLUTION", format!("{OBS_MAP_RESOLUTION:?}u")),
  ].into_iter().unzip();
  Ok(AhoCorasick::new(patterns)?.replace_all(&sources, &replace_with))
}
