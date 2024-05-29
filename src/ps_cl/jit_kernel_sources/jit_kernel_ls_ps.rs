use std::collections::HashMap;

pub fn jit_kernel_ls_ps(config_ps: &HashMap<String, String>) -> String {
  let mut sources = String::new();
  sources.push_str(r###"
kernel void kernel_ls_ps(
    const ushort i_job,
    const uint timestamp_max,
    global const Pixel *pixel,"###);
  sources.push_str(match config_ps["show_ls_ps"].as_str() {
    "gl" => r###"
    write_only image2d_t render_buffer) {"###,
    "cv" | "none" => r###"
    global float render_buffer[3][N_ROW][2 * N_COL]) {"###,
    _ => unreachable!(),
  });
  sources.push_str(r###"
  const ushort i_row = get_global_id(0);
  const ushort i_col = get_global_id(1);
  const uint timestamp_last = pixel->timestamp_last[i_row][i_col];
  const float weight_vis = exp2(-(float)(timestamp_max - timestamp_last) / VIS_HALF_LIFE);
  const float weight_ls_ps = exp2(-(float)(timestamp_max - timestamp_last) / LS_PS_HALF_LIFE);
  const float vis = 0.1f * weight_vis * pixel->vis_moving[i_row][i_col];
  const float matrix_xx = weight_ls_ps * pixel->ls_ps_moving[0][i_row][i_col];
  const float matrix_xy = weight_ls_ps * pixel->ls_ps_moving[1][i_row][i_col];
  const float matrix_xz = weight_ls_ps * pixel->ls_ps_moving[2][i_row][i_col];
  const float matrix_yy = weight_ls_ps * pixel->ls_ps_moving[3][i_row][i_col];
  const float matrix_yz = weight_ls_ps * pixel->ls_ps_moving[4][i_row][i_col];
  const float matrix_zz = weight_ls_ps * pixel->ls_ps_moving[5][i_row][i_col];
  const float adj_matrix_xx =  (matrix_yy * matrix_zz - matrix_yz * matrix_yz);
  const float adj_matrix_xy = -(matrix_xy * matrix_zz - matrix_yz * matrix_xz);
  const float adj_matrix_xz =  (matrix_xy * matrix_yz - matrix_yy * matrix_xz);
  const float adj_matrix_yy =  (matrix_xx * matrix_zz - matrix_xz * matrix_xz);
  const float adj_matrix_yz = -(matrix_xx * matrix_yz - matrix_xy * matrix_xz);
  const float adj_matrix_zz =  (matrix_xx * matrix_yy - matrix_xy * matrix_xy) + 1e-3f;
  const float3 adj_row_x = (float3)(adj_matrix_xx, adj_matrix_xy, adj_matrix_xz);
  const float3 adj_row_y = (float3)(adj_matrix_xy, adj_matrix_yy, adj_matrix_yz);
  const float3 adj_row_z = (float3)(adj_matrix_xz, adj_matrix_yz, adj_matrix_zz);
  float3 normal = (float3)(0.f, 0.f, 1.f);
  for (uint i = 0; i < 1024u; ++i) {
    normal = (float3)(dot(adj_row_x, normal), dot(adj_row_y, normal), dot(adj_row_z, normal));
    normal /= length(normal) + 1e-6f;
  }
  if (normal.z < 0.f) {
    normal = -normal;
  }
  const float4 color_event = (float4)(1.f - max(0.f, -vis), 1.f - fabs(vis), 1.f - max(0.f, vis), 1.f);
  const ushort i_col_render_buffer = i_col * N_JOBS + i_job;"###);
  sources.push_str(match config_ps["show_ls_ps"].as_str() {
    "gl" => r###"
  const float4 color_normal = (float4)(0.5f + 0.5f * normal.y, 0.5f - 0.5f * normal.x, 0.5f + 0.5f * normal.z, 1.f);
  write_imagef(render_buffer, (int2)(i_col_render_buffer, N_ROW - 1 - i_row), color_event);
  write_imagef(render_buffer, (int2)(N_COL + i_col_render_buffer, N_ROW - 1 - i_row), color_normal);"###,
    "cv" | "none" => r###"
  render_buffer[0][i_row][i_col_render_buffer] = color_event.x;
  render_buffer[1][i_row][i_col_render_buffer] = color_event.y;
  render_buffer[2][i_row][i_col_render_buffer] = color_event.z;
  render_buffer[0][i_row][N_COL + i_col_render_buffer] = normal.x;
  render_buffer[1][i_row][N_COL + i_col_render_buffer] = normal.y;
  render_buffer[2][i_row][N_COL + i_col_render_buffer] = normal.z;"###,
    _ => unreachable!(),
  });
  sources.push_str(r###"
}
"###);
  sources
}
