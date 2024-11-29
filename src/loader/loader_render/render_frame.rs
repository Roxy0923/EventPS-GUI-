use std::collections::HashMap;
use fastrand::Rng;
use anyhow::Result;
use nalgebra as na;
use ndarray::prelude::*;
use libredr::{LibreDR, camera, light_source};
use libredr_common::{render, message::*, geometry::Geometry};
use super::super::super::get_scan_pattern;

pub async fn render_frame(
    i_frame: usize,
    config_loader_render: &HashMap<String, String>,
    vertex_hash: Hash,
    transform_v: Array2<f32>,
    data_cache: DataCache) ->Result<(Array3<f32>, Array2<f32>)> {
  let mut client = LibreDR::new(config_loader_render["client_connect"].to_owned(),
                                config_loader_render["client_unix"].to_lowercase().parse::<bool>()?,
                                config_loader_render["client_tls"].to_lowercase().parse::<bool>()?).await?;
  let mut rng = Rng::with_seed(config_loader_render["seed"].parse::<u64>()?);
  let n_frames = config_loader_render["n_frames"].parse::<usize>()?;
  let duration = config_loader_render["duration"].parse::<f32>()?;
  let time = i_frame as f32 / (n_frames - 1) as f32 * duration;
  let light_dir = get_scan_pattern(&config_loader_render)?(time);
  let mut geometry = Geometry::new();
  geometry.add_vertex_hash(vertex_hash, transform_v, Array2::eye(3), &data_cache)?;
  let intrinsic = na::Matrix3::new(0.5, 0.0, 0.5,
                                   0.0, 0.5, 0.5,
                                   0.0, 0.0, 1.0);
  let extrinsic_orthogonal = camera::look_at_extrinsic(na::Vector3::new( 0., 0., 10.).as_view(),
                                                       na::Vector3::new( 0., 0.,  0.).as_view(),
                                                       na::Vector3::new(-1., 0.,  0.).as_view())?;
  let resolution = [config_loader_render["width"].parse::<usize>()?, config_loader_render["height"].parse::<usize>()?];
  let ray_orthogonal = camera::orthogonal_ray(&resolution,
                                              intrinsic.as_view(),
                                              extrinsic_orthogonal.as_view())?;
  let texture_resolution = config_loader_render["texture_resolution"].parse::<usize>()?;
  let mut texture = Array3::zeros((14, texture_resolution, texture_resolution));
  texture.slice_mut(s![3..6, .., ..]).iter_mut().for_each(|x| *x = 0.01 + 0.39 * rng.f32());
  texture.slice_mut(s![6..9, .., ..]).iter_mut().for_each(|x| *x = 0.01 + 0.09 * rng.f32());
  texture.slice_mut(s![9..10, .., ..]).iter_mut().for_each(|x| *x = 0.01 + 0.09 * rng.f32());
  let envmap = light_source::directional_envmap(256, light_dir.as_view(), 1.)?;
  let reflection_diffuse = if config_loader_render["diffuse_enable"].to_lowercase().parse::<bool>()? {
    render::REFLECTION_DIFFUSE_LAMBERTIAN
  } else {
    render::REFLECTION_DIFFUSE_NONE
  };
  let reflection_specular = if config_loader_render["specular_enable"].to_lowercase().parse::<bool>()? {
    render::REFLECTION_SPECULAR_TORRANCE_SPARROW_BECKMANN
  } else {
    render::REFLECTION_SPECULAR_NONE
  };
  let switches = (render::MISS_ENVMAP, render::REFLECTION_NORMAL_VERTEX, reflection_diffuse, reflection_specular);
  let render = client.ray_tracing_forward(&geometry,
                                          &data_cache,
                                          ray_orthogonal.into_dyn(),
                                          texture.clone(),
                                          envmap,
                                          (256, 0),
                                          (2, 0, 0, 0),
                                          switches,
                                          (1e-3, 1e-3, 1e-3),
                                          false,
                                          false,
                                          rng.i32(-65536..-1),
                                          1024).await?.into_dimensionality::<Ix3>()?;
  client.close().await?;
  let render_roughness = Array2::from_shape_fn((resolution[1], resolution[0]), |(i_row, i_col)| {
    let (u, v) = (render[(3, i_row, i_col)], render[(4, i_row, i_col)]);
    let i_row = texture_resolution - 1 - (v * texture_resolution as f32).floor() as usize;
    let i_col = (u * texture_resolution as f32).floor() as usize;
    return texture[(9, i_row.clamp(0, texture_resolution - 1), i_col.clamp(0, texture_resolution - 1))]
  });
  Ok((render, render_roughness))
}
