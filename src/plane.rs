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
  pub alloc_height: usize,
  pub width: usize,
  pub height: usize,
  pub xdec: usize,
  pub ydec: usize,
  pub xorigin: usize,
  pub yorigin: usize
}

/// Absolute offset in pixels inside a plane
#[derive(Debug)]
pub struct PlaneOffset {
  pub x: isize,
  pub y: isize
}

#[derive(Debug, Clone)]
pub struct Plane {
  pub data: Vec<u16>,
  pub cfg: PlaneConfig
}

impl Plane {
  /// Stride alignment in bytes.
  const STRIDE_ALIGNMENT_LOG2: usize = 4;

  /// Data alignment in bytes.
  const DATA_ALIGNMENT_LOG2: usize = 4;

  pub fn new(width: usize, height: usize, xdec: usize, ydec: usize, xpad: usize, ypad: usize) -> Plane {
    let xorigin = xpad.align_power_of_two(Plane::STRIDE_ALIGNMENT_LOG2 - 1);
    let yorigin = ypad;
    let stride = (xorigin + width + xpad).align_power_of_two(Plane::STRIDE_ALIGNMENT_LOG2 - 1);
    let alloc_height = yorigin + height + ypad;
    let data = vec![128u16; stride * alloc_height];
    assert!(is_aligned(data.as_ptr(), Plane::DATA_ALIGNMENT_LOG2));
    Plane { data, cfg: PlaneConfig { stride, alloc_height, width, height, xdec, ydec, xorigin, yorigin } }
  }

  pub fn pad(&mut self) {
    let xorigin = self.cfg.xorigin;
    let yorigin = self.cfg.yorigin;
    let stride = self.cfg.stride;
    let width = self.cfg.width;
    let height = self.cfg.height;


    if xorigin > 0 {
      for y in 0..height {
        let mut ps = self.mut_slice(&PlaneOffset { x: -(xorigin as isize), y: y as isize });
        let s = ps.as_mut_slice_w_width(xorigin + 1);
        let fill_val = s[xorigin];
        for val in s[..xorigin].iter_mut() { *val = fill_val; }
      }
    }

    if xorigin + width < stride {
      for y in 0..height {
        let mut ps = self.mut_slice(&PlaneOffset { x: width as isize - 1, y: y as isize });
        let s = ps.as_mut_slice_w_width(stride - xorigin - width + 1);
        let fill_val = s[0];
        for val in s[1..].iter_mut() { *val = fill_val; }
      }
    }

    if yorigin > 0 {
      let mut ps = self.mut_slice(&PlaneOffset { x: -(xorigin as isize), y: -(yorigin as isize) });
      let (s1, s2) = ps.as_mut_slice().split_at_mut(yorigin*stride);
      for y in 0 .. yorigin {
        s1[y*stride..y*stride+stride].copy_from_slice(&s2[..stride]);
      }
    }

    if yorigin + height < self.cfg.alloc_height {
      let mut ps = self.mut_slice(&PlaneOffset { x: -(xorigin as isize), y: height as isize - 1 });
      let (s2, s1) = ps.as_mut_slice().split_at_mut(stride);
      for y in 0 .. yorigin {
        s1[y*stride..y*stride+stride].copy_from_slice(&s2[..stride]);
      }
    }
  }

  pub fn slice<'a>(&'a self, po: &PlaneOffset) -> PlaneSlice<'a> {
    PlaneSlice { plane: self, x: po.x, y: po.y }
  }

  pub fn mut_slice<'a>(&'a mut self, po: &PlaneOffset) -> PlaneMutSlice<'a> {
    PlaneMutSlice { plane: self, x: po.x, y: po.y }
  }

  pub fn p(&self, x: usize, y: usize) -> u16 {
    self.data[(y + self.cfg.yorigin) * self.cfg.stride + (x + self.cfg.xorigin)]
  }

  pub fn data_origin(&self) -> &[u16] {
    &self.data[self.cfg.yorigin * self.cfg.stride + self.cfg.xorigin..]
  }

  pub fn data_origin_mut(&mut self) -> &mut [u16] {
    &mut self.data[self.cfg.yorigin * self.cfg.stride + self.cfg.xorigin..]
  }

  pub fn copy_from_raw_u8(
    &mut self, source: &[u8], source_stride: usize, source_bytewidth: usize
  ) {
    let stride = self.cfg.stride;
    for (self_row, source_row) in
      self.data_origin_mut().chunks_mut(stride).zip(source.chunks(source_stride))
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
  pub x: isize,
  pub y: isize
}

impl<'a> PlaneSlice<'a> {
  pub fn as_slice(&'a self) -> &'a [u16] {
    let stride = self.plane.cfg.stride;
    let base = (self.y + self.plane.cfg.yorigin as isize) as usize * stride + (self.x + self.plane.cfg.xorigin as isize) as usize;
    &self.plane.data[base..]
  }

  pub fn as_slice_clamped(&'a self) -> &'a [u16] {
    let stride = self.plane.cfg.stride;
    let y = (self.y.min(self.plane.cfg.height as isize) + self.plane.cfg.yorigin as isize).max(0) as usize;
    let x = (self.x.min(self.plane.cfg.width as isize) + self.plane.cfg.xorigin as isize).max(0) as usize;
    &self.plane.data[y * stride + x..]
  }

  pub fn as_slice_w_width(&'a self, width: usize) -> &'a [u16] {
    let stride = self.plane.cfg.stride;
    let base = (self.y + self.plane.cfg.yorigin as isize) as usize * stride + (self.x + self.plane.cfg.xorigin as isize) as usize;
    &self.plane.data[base .. base + width]
  }

  pub fn subslice(&'a self, xo: usize, yo: usize) -> PlaneSlice<'a> {
    PlaneSlice { plane: self.plane, x: self.x + xo as isize, y: self.y + yo as isize }
  }

  /// A slice starting i pixels above the current one.
  pub fn go_up(&'a self, i: usize) -> PlaneSlice<'a> {
    PlaneSlice { plane: self.plane, x: self.x, y: self.y - i as isize }
  }

  /// A slice starting i pixels to the left of the current one.
  pub fn go_left(&'a self, i: usize) -> PlaneSlice<'a> {
    PlaneSlice { plane: self.plane, x: self.x - i as isize, y: self.y }
  }

  pub fn p(&self, add_x: usize, add_y: usize) -> u16 {
    let new_y = (self.y + add_y as isize + self.plane.cfg.yorigin as isize) as usize;
    let new_x = (self.x + add_x as isize + self.plane.cfg.xorigin as isize) as usize;
    self.plane.data[new_y * self.plane.cfg.stride + new_x]
  }
}

pub struct PlaneMutSlice<'a> {
  pub plane: &'a mut Plane,
  pub x: isize,
  pub y: isize
}

impl<'a> PlaneMutSlice<'a> {
  pub fn as_mut_slice(&'a mut self) -> &'a mut [u16] {
    let stride = self.plane.cfg.stride;
    let base = (self.y + self.plane.cfg.yorigin as isize) as usize * stride + (self.x + self.plane.cfg.xorigin as isize) as usize;
    &mut self.plane.data[base..]
  }

  pub fn as_mut_slice_w_width(&'a mut self, width: usize) -> &'a mut [u16] {
    let stride = self.plane.cfg.stride;
    let y = self.y + self.plane.cfg.yorigin as isize;
    let x = self.x + self.plane.cfg.xorigin as isize;
    assert!(y >= 0);
    assert!(x >= 0);
    let base = y as usize * stride + x as usize;
    &mut self.plane.data[base .. base + width]
  }


  pub fn offset(&self, add_x: usize, add_y: usize) -> &[u16] {
    let new_y = (self.y + add_y as isize + self.plane.cfg.yorigin as isize) as usize;
    let new_x = (self.x + add_x as isize + self.plane.cfg.xorigin as isize) as usize;
    &self.plane.data[new_y * self.plane.cfg.stride + new_x..]
  }

  pub fn offset_as_mutable(
    &'a mut self, add_x: usize, add_y: usize
  ) -> &'a mut [u16] {
    let new_y = (self.y + add_y as isize + self.plane.cfg.yorigin as isize) as usize;
    let new_x = (self.x + add_x as isize + self.plane.cfg.xorigin as isize) as usize;
    &mut self.plane.data[new_y * self.plane.cfg.stride + new_x..]
  }

  // FIXME: code duplication with PlaneSlice

  /// A slice starting i pixels above the current one.
  pub fn go_up(&'a self, i: usize) -> PlaneSlice<'a> {
    PlaneSlice { plane: self.plane, x: self.x, y: self.y - i as isize }
  }

  /// A slice starting i pixels to the left of the current one.
  pub fn go_left(&'a self, i: usize) -> PlaneSlice<'a> {
    PlaneSlice { plane: self.plane, x: self.x - i as isize, y: self.y }
  }

  pub fn p(&self, add_x: usize, add_y: usize) -> u16 {
    let new_y = (self.y + add_y as isize + self.plane.cfg.yorigin as isize) as usize;
    let new_x = (self.x + add_x as isize + self.plane.cfg.xorigin as isize) as usize;
    self.plane.data[new_y * self.plane.cfg.stride + new_x]
  }
}
