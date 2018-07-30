// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![cfg_attr(feature = "cargo-clippy", allow(cast_lossless))]

use util::*;

/// Plane-specific configuration.
#[derive(Debug, Clone)]
pub struct PlaneConfig {
  pub stride: usize,
  pub width: usize,
  pub height: usize,
  pub xdec: usize,
  pub ydec: usize
}

/// Absolute offset in pixels inside a plane
#[derive(Debug)]
pub struct PlaneOffset {
  pub x: usize,
  pub y: usize
}

#[derive(Debug, Clone)]
pub struct Plane {
  pub data: Vec<u16>,
  pub cfg: PlaneConfig
}

impl Plane {
  pub fn new(width: usize, height: usize, xdec: usize, ydec: usize) -> Plane {
    let stride = width.align_power_of_two(4); // Force 16 byte alignment.
    Plane {
      data: vec![128; stride * height],
      cfg: PlaneConfig { stride, width, height, xdec, ydec }
    }
  }

  pub fn slice<'a>(&'a self, po: &PlaneOffset) -> PlaneSlice<'a> {
    PlaneSlice { plane: self, x: po.x, y: po.y }
  }

  pub fn mut_slice<'a>(&'a mut self, po: &PlaneOffset) -> PlaneMutSlice<'a> {
    PlaneMutSlice { plane: self, x: po.x, y: po.y }
  }

  pub fn p(&self, x: usize, y: usize) -> u16 {
    self.data[y * self.cfg.stride + x]
  }

  pub fn copy_from_raw_u8(
    &mut self, source: &[u8], source_stride: usize, source_bytewidth: usize
  ) {
    let stride = self.cfg.stride;
    for (self_row, source_row) in
      self.data.chunks_mut(stride).zip(source.chunks(source_stride))
    {
      match source_bytewidth {
        1 => for (self_pixel, source_pixel) in
          self_row.iter_mut().zip(source_row.iter())
        {
          *self_pixel = *source_pixel as u16;
        },
        2 => for (self_pixel, bytes) in
          self_row.iter_mut().zip(source_row.chunks(2))
        {
          *self_pixel = (bytes[1] as u16) << 8 | (bytes[0] as u16);
        },

        _ => {}
      }
    }
  }
}

pub struct PlaneSlice<'a> {
  pub plane: &'a Plane,
  pub x: usize,
  pub y: usize
}

impl<'a> PlaneSlice<'a> {
  pub fn as_slice(&'a self) -> &'a [u16] {
    let stride = self.plane.cfg.stride;
    &self.plane.data[self.y * stride + self.x..]
  }

  pub fn subslice(&'a self, xo: usize, yo: usize) -> PlaneSlice<'a> {
    PlaneSlice { plane: self.plane, x: self.x + xo, y: self.y + yo }
  }

  /// A slice starting i pixels above the current one.
  pub fn go_up(&'a self, i: usize) -> PlaneSlice<'a> {
    PlaneSlice { plane: self.plane, x: self.x, y: self.y - i }
  }

  /// A slice starting i pixels to the left of the current one.
  pub fn go_left(&'a self, i: usize) -> PlaneSlice<'a> {
    PlaneSlice { plane: self.plane, x: self.x - i, y: self.y }
  }

  pub fn p(&self, add_x: usize, add_y: usize) -> u16 {
    let new_y = self.y + add_y;
    let new_x = self.x + add_x;
    self.plane.data[new_y * self.plane.cfg.stride + new_x]
  }
}

pub struct PlaneMutSlice<'a> {
  pub plane: &'a mut Plane,
  pub x: usize,
  pub y: usize
}

impl<'a> PlaneMutSlice<'a> {
  pub fn as_mut_slice(&'a mut self) -> &'a mut [u16] {
    let stride = self.plane.cfg.stride;
    &mut self.plane.data[self.y * stride + self.x..]
  }

  pub fn offset(&self, add_x: usize, add_y: usize) -> &[u16] {
    let new_y = self.y + add_y;
    let new_x = self.x + add_x;
    &self.plane.data[new_y * self.plane.cfg.stride + new_x ..]
  }

  pub fn offset_as_mutable(&'a mut self, add_x: usize, add_y: usize) -> &'a mut [u16] {
    let new_y = self.y + add_y;
    let new_x = self.x + add_x;
    &mut self.plane.data[new_y * self.plane.cfg.stride + new_x ..]
  }

    // FIXME: code duplication with PlaneSlice

  /// A slice starting i pixels above the current one.
  pub fn go_up(&'a self, i: usize) -> PlaneSlice<'a> {
    PlaneSlice { plane: self.plane, x: self.x, y: self.y - i }
  }

  /// A slice starting i pixels to the left of the current one.
  pub fn go_left(&'a self, i: usize) -> PlaneSlice<'a> {
    PlaneSlice { plane: self.plane, x: self.x - i, y: self.y }
  }

  pub fn p(&self, add_x: usize, add_y: usize) -> u16 {
    let new_y = self.y + add_y;
    let new_x = self.x + add_x;
    self.plane.data[new_y * self.plane.cfg.stride + new_x]
  }
}
