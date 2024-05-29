pub fn jit_kernel_cnn_ps() -> String {
  let mut sources = String::new();
  sources.push_str(r###"
kernel void kernel_cnn_ps(
    const ushort i_job,
    const uint timestamp_max,
    global const Pixel *pixel,
    global float gather_buffer[3][OBS_MAP_RESOLUTION][OBS_MAP_RESOLUTION][N_ROW][N_COL]) {
  const ushort i_row = get_global_id(0);
  const ushort i_col = get_global_id(1);
  const ushort i_col_gather_buffer = i_col * N_JOBS + i_job;
  for (uint i_row_obs = 0; i_row_obs < OBS_MAP_RESOLUTION; ++i_row_obs) {
    for (uint i_col_obs = 0; i_col_obs < OBS_MAP_RESOLUTION; ++i_col_obs) {
      const uint obs_map_timestamp_last = pixel->obs_map_timestamp_last[i_row_obs][i_col_obs][i_row][i_col];
      const float weight = exp2(-(float)(timestamp_max - obs_map_timestamp_last) / CNN_PS_HALF_LIFE);
      for (uint i_channel = 0; i_channel < 3; ++i_channel) {
        gather_buffer[i_channel][i_row_obs][i_col_obs][i_row][i_col_gather_buffer] =
          weight * pixel->obs_map[i_channel][i_row_obs][i_col_obs][i_row][i_col];
      }
    }
  }
}
"###);
  sources
}
