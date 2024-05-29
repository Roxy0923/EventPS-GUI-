pub fn jit_kernel_ps_fcn() -> String {
  let mut sources = String::new();
  sources.push_str(r###"
kernel void kernel_ps_fcn(
    const ushort i_job,
    const uint timestamp_max,
    global const Pixel *pixel,
    global float gather_buffer[RECORD_N_BIN - 2][3][N_ROW][N_COL]) {
  const ushort i_row = get_global_id(0);
  const ushort i_col = get_global_id(1);
  const ushort i_col_gather_buffer = i_col * N_JOBS + i_job;
  const uint timestamp_last = pixel->timestamp_last[i_row][i_col];
  const uint i_bin_last = timestamp_last / TIME_PER_BIN % RECORD_N_BIN;
  const uint i_bin_max = timestamp_max / TIME_PER_BIN % RECORD_N_BIN;
  for (uint i_bin_add = 2; i_bin_add < RECORD_N_BIN; ++i_bin_add) {
    float3 null_vector;
    if (i_bin_add < (i_bin_max - i_bin_last + 1 + RECORD_N_BIN) % RECORD_N_BIN) {
      null_vector = (float3)(0.f, 0.f, 0.f);
    } else {
      const uint i_bin = (i_bin_max - i_bin_add + RECORD_N_BIN) % RECORD_N_BIN;
      null_vector = (float3)(pixel->null_vector_bin[i_bin][0][i_row][i_col],
                             pixel->null_vector_bin[i_bin][1][i_row][i_col],
                             pixel->null_vector_bin[i_bin][2][i_row][i_col]);
    }
    gather_buffer[i_bin_add - 2][0][i_row][i_col_gather_buffer] = null_vector.x;
    gather_buffer[i_bin_add - 2][1][i_row][i_col_gather_buffer] = null_vector.y;
    gather_buffer[i_bin_add - 2][2][i_row][i_col_gather_buffer] = null_vector.z;
  }
}
"###);
  sources
}
