use std::fs::File;
use std::io::Write;
use std::collections::HashMap;
use tracing::info;
use ndarray::prelude::*;
use anyhow::{Result, anyhow, ensure};

pub struct VideoWriter {
  context_frame: rav1e::Context<u16>,
  writer_frame: File,
  context_normal: rav1e::Context<u16>,
  writer_normal: File,
}

impl VideoWriter {
  pub fn new(config_loader_render: &HashMap<String, String>) -> Result<Self> {
    let save_video: [&str; 2] = config_loader_render["save_video"]
      .split([' ', '|'])
      .filter(|v| !v.is_empty())
      .collect::<Vec<&str>>()
      .try_into()
      .map_err(|_| anyhow!("loader_writer::VideoWriter::new: `save_video` should be two path for frame and normal"))?;
    let width = config_loader_render["width"].parse::<usize>()?;
    let height = config_loader_render["height"].parse::<usize>()?;
    let encoder_config = rav1e::EncoderConfig {
      width,
      height,
      time_base: rav1e::data::Rational{
        num: 1,
        den: 10,
      },
      bit_depth: 10,
      chroma_sampling: rav1e::color::ChromaSampling::Cs444,
      pixel_range: rav1e::color::PixelRange::Full,
      color_description: Some(rav1e::color::ColorDescription {
        color_primaries: rav1e::color::ColorPrimaries::BT709,
        transfer_characteristics: rav1e::color::TransferCharacteristics::SRGB,
        matrix_coefficients: rav1e::color::MatrixCoefficients::Identity,
      }),
      // `quantizer` is the lower the better
      quantizer: 8, // 99 quality
      // quantizer: 38, // 95 quality
      // quantizer: 76, // 90 quality
      speed_settings: rav1e::config::SpeedSettings::from_preset(6),
      ..Default::default()
    };
    let config = rav1e::Config::default()
      .with_encoder_config(encoder_config)
      .with_threads(4);
    info!("loader_render::VideoWriter::new: Creating context");
    Ok(VideoWriter {
      context_frame: config.new_context()?,
      writer_frame: File::create(save_video[0])?,
      context_normal: config.new_context()?,
      writer_normal: File::create(save_video[1])?,
    })
  }

  fn rav1e_send_frame(context: &mut rav1e::Context<u16>, image: ArrayView3<u16>, writer: &mut File) -> Result<usize> {
    let mut frame = context.new_frame();
    let image = [image.index_axis(Axis(0), 1), image.index_axis(Axis(0), 0), image.index_axis(Axis(0), 2)];
    for (plane, image) in frame.planes.iter_mut().zip(image.into_iter()) {
      ensure!(image.shape() == [plane.cfg.height, plane.cfg.width]);
      let plane_stride = plane.cfg.stride;
      let plane_iter = plane.data_origin_mut().chunks_exact_mut(plane_stride);
      for (plane_row, image_row) in plane_iter.zip(image.rows()) {
        plane_row[..image_row.len()].copy_from_slice(image_row.as_slice().expect("Internal created"));
      }
    }
    context.send_frame(frame)?;
    let mut written_bytes = 0;
    loop {
      match context.receive_packet() {
        Ok(packet) => {
          written_bytes += packet.data.len();
          writer.write_all(&packet.data)?
        },
        Err(rav1e::EncoderStatus::NeedMoreData) => break,
        Err(rav1e::EncoderStatus::Encoded) => break,
        Err(e) => return Err(e.into()),
      };
    }
    Ok(written_bytes)
  }

  pub fn add_frame(&mut self, frame: ArrayView3<f32>, normal: ArrayView3<f32>) -> Result<()> {
    let frame = crate::map_image(frame)?.map(|x| (x * 1024.) as u16);
    let bytes_frame = Self::rav1e_send_frame(&mut self.context_frame, frame.view(), &mut self.writer_frame)?;
    let normal = crate::map_normal(normal)?.map(|x| (x * 1024.) as u16);
    let bytes_normal = Self::rav1e_send_frame(&mut self.context_normal, normal.view(), &mut self.writer_normal)?;
    info!("loader_render::VideoWriter::add_frame: Written frame {bytes_frame} bytes, normal {bytes_normal} bytes");
    Ok(())
  }

  fn rav1e_flush(context: &mut rav1e::Context<u16>, writer: &mut File) -> Result<usize> {
    context.flush();
    let mut written_bytes = 0;
    loop {
      match context.receive_packet() {
        Ok(packet) => {
          written_bytes += packet.data.len();
          writer.write_all(&packet.data)?
        },
        Err(rav1e::EncoderStatus::Encoded) => continue,
        Err(rav1e::EncoderStatus::LimitReached) => break,
        Err(e) => return Err(e.into()),
      };
    }
    Ok(written_bytes)
  }

  pub fn flush(&mut self) -> Result<()> {
    let bytes_frame = Self::rav1e_flush(&mut self.context_frame, &mut self.writer_frame)?;
    let bytes_normal = Self::rav1e_flush(&mut self.context_normal, &mut self.writer_normal)?;
    info!("loader_render::VideoWriter::flush: Written frame {bytes_frame} bytes, normal {bytes_normal} bytes");
    Ok(())
  }
}
