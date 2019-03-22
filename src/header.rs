// Copyright (c) 2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::api::*;
use crate::context::*;
use crate::ec::*;
use crate::lrf::*;
use crate::partition::*;
use crate::util::Pixel;

use crate::SegmentationState;
use crate::DeblockState;
use crate::FrameState;
use crate::FrameInvariants;
use crate::Sequence;

use bitstream_io::{BitWriter, BigEndian};

use std;
use std::io;

pub const PRIMARY_REF_NONE: u32 = 7;
pub const ALL_REF_FRAMES_MASK: u32 = (1 << REF_FRAMES) - 1;

const PRIMARY_REF_BITS: u32 = 3;

#[allow(unused)]
const OP_POINTS_IDC_BITS: usize = 12;
#[allow(unused)]
const LEVEL_MAJOR_MIN: usize = 2;
#[allow(unused)]
const LEVEL_MAJOR_BITS: usize = 3;
#[allow(unused)]
const LEVEL_MINOR_BITS: usize = 2;
#[allow(unused)]
const LEVEL_BITS: usize = LEVEL_MAJOR_BITS + LEVEL_MINOR_BITS;
const FRAME_ID_LENGTH: usize = 15;
const DELTA_FRAME_ID_LENGTH: usize = 14;


#[allow(dead_code,non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReferenceMode {
  SINGLE = 0,
  COMPOUND = 1,
  SELECT = 2,
}

#[allow(non_camel_case_types)]
pub enum ObuType {
  OBU_SEQUENCE_HEADER = 1,
  OBU_TEMPORAL_DELIMITER = 2,
  OBU_FRAME_HEADER = 3,
  OBU_TILE_GROUP = 4,
  OBU_METADATA = 5,
  OBU_FRAME = 6,
  OBU_REDUNDANT_FRAME_HEADER = 7,
  OBU_TILE_LIST = 8,
  OBU_PADDING = 15,
}

#[derive(Clone,Copy)]
#[allow(non_camel_case_types)]
pub enum ObuMetaType {
  OBU_META_HDR_CLL = 1,
  OBU_META_HDR_MDCV = 2,
  OBU_META_SCALABILITY = 3,
  OBU_META_ITUT_T35 = 4,
  OBU_META_TIMECODE = 5,
}

impl ObuMetaType {
  fn size(self) -> u64 {
    use self::ObuMetaType::*;
    match self {
      OBU_META_HDR_CLL => 4,
      OBU_META_HDR_MDCV => 24,
      _ => 0,
    }
  }
}

pub trait ULEB128Writer {
  fn write_uleb128(&mut self, payload: u64) -> io::Result<()>;
}

impl<W: io::Write> ULEB128Writer for BitWriter<W, BigEndian> {
  fn write_uleb128(&mut self, payload: u64) -> io::Result<()> {
    // NOTE from libaom:
    // Disallow values larger than 32-bits to ensure consistent behavior on 32 and
    // 64 bit targets: value is typically used to determine buffer allocation size
    // when decoded.
    fn uleb_size_in_bytes(mut value: u64) -> usize {
      let mut size = 0;
      loop {
        size += 1;
        value >>= 7;
        if value == 0 { break; }
      }
      size
    }

    fn uleb_encode(mut value: u64, coded_value: &mut [u8]) -> usize {
      let leb_size = uleb_size_in_bytes(value);

      for i in 0..leb_size {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 { byte |= 0x80 };  // Signal that more bytes follow.
        coded_value[i] = byte;
      }

      leb_size
    }

    let mut coded_payload_length = [0 as u8; 8];
    let leb_size = uleb_encode(payload, &mut coded_payload_length);
    for i in 0..leb_size {
      self.write(8, coded_payload_length[i])?;
    }
    Ok(())
  }
}

pub trait UncompressedHeader {
  // Start of OBU Headers
  fn write_obu_header(
    &mut self, obu_type: ObuType, obu_extension: u32
  ) -> io::Result<()>;
  fn write_metadata_obu(
    &mut self, obu_meta_type: ObuMetaType, seq: Sequence
  ) -> io::Result<()>;
  fn write_sequence_header_obu<T: Pixel>(
    &mut self, fi: &mut FrameInvariants<T>
  ) -> io::Result<()>;
  fn write_frame_header_obu<T: Pixel>(
    &mut self, fi: &FrameInvariants<T>, fs: &FrameState<T>
  ) -> io::Result<()>;
  fn write_sequence_header<T: Pixel>(
    &mut self, fi: &mut FrameInvariants<T>
  ) -> io::Result<()>;
  fn write_color_config(
    &mut self, seq: &mut Sequence
  ) -> io::Result<()>;
  // End of OBU Headers

  fn write_frame_size<T: Pixel>(
    &mut self, fi: &FrameInvariants<T>
  ) -> io::Result<()>;
  fn write_deblock_filter_a<T: Pixel>(
    &mut self, fi: &FrameInvariants<T>, deblock: &DeblockState
  ) -> io::Result<()>;
  fn write_deblock_filter_b<T: Pixel>(
    &mut self, fi: &FrameInvariants<T>, deblock: &DeblockState
  ) -> io::Result<()>;
  fn write_frame_cdef<T: Pixel>(
    &mut self, fi: &FrameInvariants<T>
  ) -> io::Result<()>;
  fn write_frame_lrf<T: Pixel>(
    &mut self, fi: &FrameInvariants<T>, rs: &RestorationState
  ) -> io::Result<()>;
  fn write_segment_data<T: Pixel>(
    &mut self, fi: &FrameInvariants<T>, segmentation: &SegmentationState
  ) -> io::Result<()>;
  fn write_delta_q(
    &mut self, delta_q: i8
  ) -> io::Result<()>;
}


impl<W: io::Write> UncompressedHeader for BitWriter<W, BigEndian> {
  // Start of OBU Headers
  // Write OBU Header syntax
  fn write_obu_header(
    &mut self, obu_type: ObuType, obu_extension: u32
  ) -> io::Result<()> {
    self.write_bit(false)?; // forbidden bit.
    self.write(4, obu_type as u32)?;
    self.write_bit(obu_extension != 0)?;
    self.write_bit(true)?; // obu_has_payload_length_field
    self.write_bit(false)?; // reserved

    if obu_extension != 0 {
      unimplemented!();
      //self.write(8, obu_extension & 0xFF)?; size += 8;
    }

    Ok(())
  }

  fn write_metadata_obu(
    &mut self, obu_meta_type: ObuMetaType, seq: Sequence
  ) -> io::Result<()> {
    // header
    self.write_obu_header(ObuType::OBU_METADATA, 0)?;

    // uleb128() - length
    // we use a constant value to avoid computing the OBU size every time
    // since it is fixed (depending on the metadata)
    // +2 is for the metadata_type field and the trailing bits byte
    self.write_uleb128(obu_meta_type.size() + 2)?;

    // uleb128() - metadata_type (1 byte)
    self.write_uleb128(obu_meta_type as u64)?;

    match obu_meta_type {
      ObuMetaType::OBU_META_HDR_CLL => {
        let cll = seq.content_light.unwrap();
        self.write(16, cll.max_content_light_level)?;
        self.write(16, cll.max_frame_average_light_level)?;
      },
      ObuMetaType::OBU_META_HDR_MDCV => {
        let mdcv = seq.mastering_display.unwrap();
        for i in 0..3 {
          self.write(16, mdcv.primaries[i].x)?;
          self.write(16, mdcv.primaries[i].y)?;
        }

        self.write(16, mdcv.white_point.x)?;
        self.write(16, mdcv.white_point.y)?;

        self.write(32, mdcv.max_luminance)?;
        self.write(32, mdcv.min_luminance)?;
      },
      _ => {}
    }

    // trailing bits (1 byte)
    self.write_bit(true)?;
    self.byte_align()?;

    Ok(())
  }

  fn write_sequence_header_obu<T: Pixel>(
    &mut self, fi: &mut FrameInvariants<T>
  ) -> io::Result<()> {
    self.write(3, fi.sequence.profile)?; // profile
    self.write_bit(false)?; // still_picture
    self.write_bit(false)?; // reduced_still_picture_header
    self.write_bit(false)?; // timing info present
    self.write_bit(false)?; // initial display delay present flag
    self.write(5, 0)?; // one operating point
    self.write(12, 0)?; // idc
    self.write(5, 31)?; // level
    self.write(1, 0)?; // tier
    if fi.sequence.reduced_still_picture_hdr {
      unimplemented!();
    }

    self.write_sequence_header(fi)?;

    self.write_color_config(&mut fi.sequence)?;

    self.write_bit(fi.sequence.film_grain_params_present)?;

    Ok(())
  }

  fn write_sequence_header<T: Pixel>(
    &mut self, fi: &mut FrameInvariants<T>
  ) -> io::Result<()> {
    self.write_frame_size(fi)?;

    let seq = &mut fi.sequence;

    seq.frame_id_numbers_present_flag = false;

    if !seq.reduced_still_picture_hdr {
      self.write_bit(seq.frame_id_numbers_present_flag)?;
    }

    seq.frame_id_length = FRAME_ID_LENGTH as u32;
    seq.delta_frame_id_length = DELTA_FRAME_ID_LENGTH as u32;

    if seq.frame_id_numbers_present_flag {
      // We must always have delta_frame_id_length < frame_id_length,
      // in order for a frame to be referenced with a unique delta.
      // Avoid wasting bits by using a coding that enforces this restriction.
      self.write(4, seq.delta_frame_id_length - 2)?;
      self.write(3, seq.frame_id_length - seq.delta_frame_id_length - 1)?;
    }

    self.write_bit(seq.use_128x128_superblock)?;
    self.write_bit(true)?; // enable filter intra
    self.write_bit(seq.enable_intra_edge_filter)?;

    if !seq.reduced_still_picture_hdr {
      self.write_bit(seq.enable_interintra_compound)?;
      self.write_bit(seq.enable_masked_compound)?;
      self.write_bit(seq.enable_warped_motion)?;
      self.write_bit(seq.enable_dual_filter)?;
      self.write_bit(seq.enable_order_hint)?;

      if seq.enable_order_hint {
        self.write_bit(seq.enable_jnt_comp)?;
        self.write_bit(seq.enable_ref_frame_mvs)?;
      }

      if seq.force_screen_content_tools == 2 {
        self.write_bit(true)?;
      } else {
        self.write_bit(false)?;
        self.write_bit(seq.force_screen_content_tools != 0)?;
      }
      if seq.force_screen_content_tools > 0 {
        if seq.force_integer_mv == 2 {
          self.write_bit(true)?;
        } else {
          self.write_bit(false)?;
          self.write_bit(seq.force_integer_mv != 0)?;
        }
      } else {
        assert!(seq.force_integer_mv == 2);
      }
      if seq.enable_order_hint {
        self.write(3, seq.order_hint_bits_minus_1)?;
      }
    }

    self.write_bit(seq.enable_superres)?;
    self.write_bit(seq.enable_cdef)?;
    self.write_bit(seq.enable_restoration)?;

    Ok(())
  }

  fn write_color_config(&mut self, seq: &mut Sequence) -> io::Result<()> {
    let high_bd = seq.bit_depth > 8;

    self.write_bit(high_bd)?;

    if seq.profile == 2 && high_bd {
      self.write_bit(seq.bit_depth == 12)?;
    }

    let monochrome = seq.chroma_sampling == ChromaSampling::Cs400;
    if seq.profile == 1 {
      assert!(!monochrome);
    } else {
      self.write_bit(monochrome)?;
    }

    if monochrome {
      unimplemented!();
    }

    // color description present
    self.write_bit(seq.color_description.is_some())?;

    let mut write_color_range = true;

    if let Some(color_description) = seq.color_description {
      self.write(8, color_description.color_primaries as u8)?;
      self.write(8, color_description.transfer_characteristics as u8)?;
      self.write(8, color_description.matrix_coefficients as u8)?;

      if color_description.color_primaries == ColorPrimaries::BT709 &&
        color_description.transfer_characteristics == TransferCharacteristics::SRGB &&
        color_description.matrix_coefficients == MatrixCoefficients::Identity {
        write_color_range = false;
        assert!(seq.chroma_sampling == ChromaSampling::Cs444);
      }
    }

    if write_color_range {
      self.write_bit(seq.pixel_range == PixelRange::Full)?; // full color range

      if monochrome {
        return Ok(());
      }

      let subsampling_x = seq.chroma_sampling != ChromaSampling::Cs444;
      let subsampling_y = seq.chroma_sampling == ChromaSampling::Cs420;

      if seq.profile == 0 {
        assert!(seq.chroma_sampling == ChromaSampling::Cs420);
      } else if seq.profile == 1 {
        assert!(seq.chroma_sampling == ChromaSampling::Cs444);
      } else {
        if seq.bit_depth == 12 {
          self.write_bit(subsampling_x)?;

          if subsampling_x {
            self.write_bit(subsampling_y)?;
          }
        } else {
          assert!(seq.chroma_sampling == ChromaSampling::Cs422);
        }
      }

      if seq.chroma_sampling == ChromaSampling::Cs420 {
        self.write(2, seq.chroma_sample_position as u32)?;
      }
    }

    self.write_bit(seq.separate_uv_delta_q)?;

    Ok(())
  }

  #[allow(unused)]
  fn write_frame_header_obu<T: Pixel>(
    &mut self, fi: &FrameInvariants<T>, fs: &FrameState<T>
  ) -> io::Result<()> {
    if fi.sequence.reduced_still_picture_hdr {
      assert!(fi.show_existing_frame);
      assert!(fi.frame_type == FrameType::KEY);
      assert!(fi.show_frame);
    } else {
      self.write_bit(fi.show_existing_frame)?;

      if fi.show_existing_frame {
        self.write(3, fi.frame_to_show_map_idx)?;

        //TODO:
        /* temporal_point_info();
        if fi.sequence.decoder_model_info_present_flag &&
           timing_info.equal_picture_interval == 0 {
          // write frame_presentation_delay;
        }
        if fi.sequence.frame_id_numbers_present_flag {
          // write display_frame_id;
        }*/

        self.write_bit(true)?; // trailing bit
        self.byte_align()?;
        return Ok(());
      }

      self.write(2, fi.frame_type as u32)?;
      self.write_bit(fi.show_frame)?; // show frame

      if fi.show_frame {
        //TODO:
        /* temporal_point_info();
        if fi.sequence.decoder_model_info_present_flag &&
           timing_info.equal_picture_interval == 0 {
          // write frame_presentation_delay;
        }*/
      } else {
        self.write_bit(fi.showable_frame)?;
      }

      if fi.frame_type == FrameType::SWITCH {
        assert!(fi.error_resilient);
      } else {
        if !(fi.frame_type == FrameType::KEY && fi.show_frame) {
          self.write_bit(fi.error_resilient)?; // error resilient
        }
      }
    }

    self.write_bit(fi.disable_cdf_update)?;

    if fi.sequence.force_screen_content_tools == 2 {
      self.write_bit(fi.allow_screen_content_tools != 0)?;
    } else {
      assert!(
        fi.allow_screen_content_tools
          == fi.sequence.force_screen_content_tools
      );
    }

    if fi.allow_screen_content_tools == 2 {
      if fi.sequence.force_integer_mv == 2 {
        self.write_bit(fi.force_integer_mv != 0)?;
      } else {
        assert!(fi.force_integer_mv == fi.sequence.force_integer_mv);
      }
    } else {
      assert!(
        fi.allow_screen_content_tools
          == fi.sequence.force_screen_content_tools
      );
    }

    if fi.sequence.frame_id_numbers_present_flag {
      unimplemented!();

      //TODO:
      //let frame_id_len = fi.sequence.frame_id_length;
      //self.write(frame_id_len, fi.current_frame_id);
    }

    let mut frame_size_override_flag = false;
    if fi.frame_type == FrameType::SWITCH {
      frame_size_override_flag = true;
    } else if fi.sequence.reduced_still_picture_hdr {
      frame_size_override_flag = false;
    } else {
      self.write_bit(frame_size_override_flag)?; // frame size overhead flag
    }

    if fi.sequence.enable_order_hint {
      let n = fi.sequence.order_hint_bits_minus_1 + 1;
      let mask = (1 << n) - 1;
      self.write(n, fi.order_hint & mask)?;
    }

    if fi.error_resilient || fi.intra_only {
    } else {
      self.write(PRIMARY_REF_BITS, fi.primary_ref_frame)?;
    }

    if fi.sequence.decoder_model_info_present_flag {
      unimplemented!();
    }

    if fi.frame_type == FrameType::KEY {
      if !fi.show_frame {
        // unshown keyframe (forward keyframe)
        unimplemented!();
        self.write(REF_FRAMES as u32, fi.refresh_frame_flags)?;
      } else {
        assert!(fi.refresh_frame_flags == ALL_REF_FRAMES_MASK);
      }
    } else {
      // Inter frame info goes here
      if fi.intra_only {
        assert!(fi.refresh_frame_flags != ALL_REF_FRAMES_MASK);
        self.write(REF_FRAMES as u32, fi.refresh_frame_flags)?;
      } else {
        // TODO: This should be set once inter mode is used
        self.write(REF_FRAMES as u32, fi.refresh_frame_flags)?;
      }
    };

    if (!fi.intra_only || fi.refresh_frame_flags != ALL_REF_FRAMES_MASK) {
      // Write all ref frame order hints if error_resilient_mode == 1
      if (fi.error_resilient && fi.sequence.enable_order_hint) {
        unimplemented!();
        //for _ in 0..REF_FRAMES {
        //  self.write(order_hint_bits_minus_1,ref_order_hint[i])?; // order_hint
        //}
      }
    }

    // if KEY or INTRA_ONLY frame
    // FIXME: Not sure whether putting frame/render size here is good idea
    if fi.intra_only {
      if frame_size_override_flag {
        unimplemented!();
      }
      if fi.sequence.enable_superres {
        unimplemented!();
      }
      self.write_bit(false)?; // render_and_frame_size_different
                              // if render_and_frame_size_different { }
      if fi.allow_screen_content_tools != 0 { // TODO: && UpscaledWidth == FrameWidth.
        self.write_bit(fi.allow_intrabc)?;
      }
    }

    let frame_refs_short_signaling = false;
    if fi.frame_type == FrameType::KEY {
      // Done by above
    } else {
      if fi.intra_only {
        // Done by above
      } else {
        if fi.sequence.enable_order_hint {
          self.write_bit(frame_refs_short_signaling)?;
          if frame_refs_short_signaling {
            unimplemented!();
          }
        }

        for i in 0..INTER_REFS_PER_FRAME {
          if !frame_refs_short_signaling {
            self.write(REF_FRAMES_LOG2 as u32, fi.ref_frames[i] as u8)?;
          }
          if fi.sequence.frame_id_numbers_present_flag {
            unimplemented!();
          }
        }
        if fi.error_resilient && frame_size_override_flag {
          unimplemented!();
        } else {
          if frame_size_override_flag {
            unimplemented!();
          }
          if fi.sequence.enable_superres {
            unimplemented!();
          }
          self.write_bit(false)?; // render_and_frame_size_different
        }
        if fi.force_integer_mv != 0 {
        } else {
          self.write_bit(fi.allow_high_precision_mv);
        }
        self.write_bit(fi.is_filter_switchable)?;
        self.write_bit(fi.is_motion_mode_switchable)?;
        self.write(2, 0)?; // EIGHTTAP_REGULAR
        if fi.error_resilient || !fi.sequence.enable_ref_frame_mvs {
        } else {
          self.write_bit(fi.use_ref_frame_mvs)?;
        }
      }
    }

    if !fi.sequence.reduced_still_picture_hdr && !fi.disable_cdf_update {
      self.write_bit(fi.disable_frame_end_update_cdf)?;
    }

    // tile
    self.write_bit(true)?; // uniform_tile_spacing_flag
    if fi.width > 64 {
      // TODO: if tile_cols > 1, write more increment_tile_cols_log2 bits
      self.write_bit(false)?; // tile cols
    }
    if fi.height > 64 {
      // TODO: if tile_rows > 1, write increment_tile_rows_log2 bits
      self.write_bit(false)?; // tile rows
    }
    // TODO: if tile_cols * tile_rows > 1 {
    // write context_update_tile_id and tile_size_bytes_minus_1 }

    // quantization
    assert!(fi.base_q_idx > 0);
    self.write(8, fi.base_q_idx)?; // base_q_idx
    self.write_delta_q(fi.dc_delta_q[0])?;
    assert!(fi.ac_delta_q[0] == 0);
    let diff_uv_delta = fi.sequence.separate_uv_delta_q
      && (fi.dc_delta_q[1] != fi.dc_delta_q[2]
          || fi.ac_delta_q[1] != fi.ac_delta_q[2]);
    if fi.sequence.separate_uv_delta_q {
      self.write_bit(diff_uv_delta)?;
    } else {
      assert!(fi.dc_delta_q[1] == fi.dc_delta_q[2]);
      assert!(fi.ac_delta_q[1] == fi.ac_delta_q[2]);
    }
    self.write_delta_q(fi.dc_delta_q[1])?;
    self.write_delta_q(fi.ac_delta_q[1])?;
    if diff_uv_delta {
      self.write_delta_q(fi.dc_delta_q[2])?;
      self.write_delta_q(fi.ac_delta_q[2])?;
    }
    self.write_bit(false)?; // no qm

    // segmentation
    self.write_segment_data(fi, &fs.segmentation)?;

    // delta_q
    self.write_bit(false)?; // delta_q_present_flag: no delta q

    // delta_lf_params in the spec
    self.write_deblock_filter_a(fi, &fs.deblock)?;

    // code for features not yet implemented....

    // loop_filter_params in the spec
    self.write_deblock_filter_b(fi, &fs.deblock)?;

    // cdef
    self.write_frame_cdef(fi)?;

    // loop restoration
    self.write_frame_lrf(fi, &fs.restoration)?;

    self.write_bit(fi.tx_mode_select)?; // tx mode

    let mut reference_select = false;
    if !fi.intra_only {
      reference_select = fi.reference_mode != ReferenceMode::SINGLE;
      self.write_bit(reference_select)?;
    }

    let skip_mode_allowed =
      fi.sequence.get_skip_mode_allowed(fi, reference_select);
    if skip_mode_allowed {
      self.write_bit(false)?; // skip_mode_present
    }

    if fi.intra_only || fi.error_resilient || !fi.sequence.enable_warped_motion
    {
    } else {
      self.write_bit(fi.allow_warped_motion)?; // allow_warped_motion
    }

    self.write_bit(fi.use_reduced_tx_set)?; // reduced tx

    // global motion
    if !fi.intra_only {
      for i in LAST_FRAME..=ALTREF_FRAME {
        let mode = fi.globalmv_transformation_type[i];
        self.write_bit(mode != GlobalMVMode::IDENTITY)?;
        if mode != GlobalMVMode::IDENTITY {
          self.write_bit(mode == GlobalMVMode::ROTZOOM)?;
          if mode != GlobalMVMode::ROTZOOM {
            self.write_bit(mode == GlobalMVMode::TRANSLATION)?;
          }
        }
        match mode {
          GlobalMVMode::IDENTITY => { /* Nothing to do */ }
          GlobalMVMode::TRANSLATION => {
            let mv_x = 0;
            let mv_x_ref = 0;
            let mv_y = 0;
            let mv_y_ref = 0;
            let bits = 12 - 6 + 3 - !fi.allow_high_precision_mv as u8;
            let bits_diff = 12 - 3 + fi.allow_high_precision_mv as u8;
            BCodeWriter::write_s_refsubexpfin(
              self,
              (1 << bits) + 1,
              3,
              mv_x_ref >> bits_diff,
              mv_x >> bits_diff
            )?;
            BCodeWriter::write_s_refsubexpfin(
              self,
              (1 << bits) + 1,
              3,
              mv_y_ref >> bits_diff,
              mv_y >> bits_diff
            )?;
          }
          GlobalMVMode::ROTZOOM => unimplemented!(),
          GlobalMVMode::AFFINE => unimplemented!()
        };
      }
    }

    if fi.sequence.film_grain_params_present && fi.show_frame {
      unimplemented!();
    }

    if fi.large_scale_tile {
      unimplemented!();
    }
    self.write_bit(true)?; // trailing bit
    self.byte_align()?;

    Ok(())
  }
  // End of OBU Headers

  fn write_frame_size<T: Pixel>(&mut self, fi: &FrameInvariants<T>) -> io::Result<()> {
    // width_bits and height_bits will have to be moved to the sequence header OBU
    // when we add support for it.
    let width_bits = 32 - (fi.width as u32).leading_zeros();
    let height_bits = 32 - (fi.height as u32).leading_zeros();
    assert!(width_bits <= 16);
    assert!(height_bits <= 16);
    self.write(4, width_bits - 1)?;
    self.write(4, height_bits - 1)?;
    self.write(width_bits, (fi.width - 1) as u16)?;
    self.write(height_bits, (fi.height - 1) as u16)?;
    Ok(())
  }

  fn write_deblock_filter_a<T: Pixel>(
    &mut self, fi: &FrameInvariants<T>, deblock: &DeblockState
  ) -> io::Result<()> {
    if fi.delta_q_present {
      if !fi.allow_intrabc {
        self.write_bit(deblock.block_deltas_enabled)?;
      }
      if deblock.block_deltas_enabled {
        self.write(2, deblock.block_delta_shift)?;
        self.write_bit(deblock.block_delta_multi)?;
      }
    }
    Ok(())
  }

  fn write_deblock_filter_b<T: Pixel>(
    &mut self, fi: &FrameInvariants<T>, deblock: &DeblockState
  ) -> io::Result<()> {
    assert!(deblock.levels[0] < 64);
    self.write(6, deblock.levels[0])?; // loop deblocking filter level 0
    assert!(deblock.levels[1] < 64);
    self.write(6, deblock.levels[1])?; // loop deblocking filter level 1
    if PLANES > 1 && (deblock.levels[0] > 0 || deblock.levels[1] > 0) {
      assert!(deblock.levels[2] < 64);
      self.write(6, deblock.levels[2])?; // loop deblocking filter level 2
      assert!(deblock.levels[3] < 64);
      self.write(6, deblock.levels[3])?; // loop deblocking filter level 3
    }
    self.write(3, deblock.sharpness)?; // deblocking filter sharpness
    self.write_bit(deblock.deltas_enabled)?; // loop deblocking filter deltas enabled
    if deblock.deltas_enabled {
      self.write_bit(deblock.delta_updates_enabled)?; // deltas updates enabled
      if deblock.delta_updates_enabled {
        // conditionally write ref delta updates
        let prev_ref_deltas = if fi.primary_ref_frame == PRIMARY_REF_NONE {
          [1, 0, 0, 0, 0, -1, -1, -1]
        } else {
          fi.rec_buffer.deblock
            [fi.ref_frames[fi.primary_ref_frame as usize] as usize]
            .ref_deltas
        };
        for i in 0..REF_FRAMES {
          let update = deblock.ref_deltas[i] != prev_ref_deltas[i];
          self.write_bit(update)?;
          if update {
            self.write_signed(7, deblock.ref_deltas[i])?;
          }
        }
        // conditionally write mode delta updates
        let prev_mode_deltas = if fi.primary_ref_frame == PRIMARY_REF_NONE {
          [0, 0]
        } else {
          fi.rec_buffer.deblock
            [fi.ref_frames[fi.primary_ref_frame as usize] as usize]
            .mode_deltas
        };
        for i in 0..2 {
          let update = deblock.mode_deltas[i] != prev_mode_deltas[i];
          self.write_bit(update)?;
          if update {
            self.write_signed(7, deblock.mode_deltas[i])?;
          }
        }
      }
    }
    Ok(())
  }

  fn write_frame_cdef<T: Pixel>(&mut self, fi: &FrameInvariants<T>) -> io::Result<()> {
    if fi.sequence.enable_cdef {
      assert!(fi.cdef_damping >= 3);
      assert!(fi.cdef_damping <= 6);
      self.write(2, fi.cdef_damping - 3)?;
      assert!(fi.cdef_bits < 4);
      self.write(2, fi.cdef_bits)?; // cdef bits
      for i in 0..(1 << fi.cdef_bits) {
        let j = i << (3 - fi.cdef_bits);
        assert!(fi.cdef_y_strengths[j] < 64);
        assert!(fi.cdef_uv_strengths[j] < 64);
        self.write(6, fi.cdef_y_strengths[j])?; // cdef y strength
        self.write(6, fi.cdef_uv_strengths[j])?; // cdef uv strength
      }
    }
    Ok(())
  }

  fn write_frame_lrf<T: Pixel>(
    &mut self, fi: &FrameInvariants<T>, rs: &RestorationState
  ) -> io::Result<()> {
    if fi.sequence.enable_restoration && !fi.allow_intrabc {
      // && !self.lossless
      let mut use_lrf = false;
      let mut use_chroma_lrf = false;
      for i in 0..PLANES {
        self.write(2, rs.planes[i].lrf_type)?; // filter type by plane
        if rs.planes[i].lrf_type != RESTORE_NONE {
          use_lrf = true;
          if i > 0 {
            use_chroma_lrf = true;
          }
        }
      }
      if use_lrf {
        // The Y shift value written here indicates shift up from superblock size
        if !fi.sequence.use_128x128_superblock {
          self.write(1, if rs.planes[0].unit_size > 64 { 1 } else { 0 })?;
        }
        if rs.planes[0].unit_size > 64 {
          self.write(1, if rs.planes[0].unit_size > 128 { 1 } else { 0 })?;
        }

        if use_chroma_lrf {
          if fi.sequence.chroma_sampling == ChromaSampling::Cs420 {
            self.write(
              1,
              if rs.planes[0].unit_size > rs.planes[1].unit_size {
                1
              } else {
                0
              }
            )?;
          }
        }
      }
    }
    Ok(())
  }

  fn write_segment_data<T: Pixel>(
    &mut self, fi: &FrameInvariants<T>, segmentation: &SegmentationState
  ) -> io::Result<()> {
    self.write_bit(segmentation.enabled)?;
    if segmentation.enabled {
      if fi.primary_ref_frame == PRIMARY_REF_NONE {
        assert_eq!(segmentation.update_map, true);
        assert_eq!(segmentation.update_data, true);
      } else {
        self.write_bit(segmentation.update_map)?;
        if segmentation.update_map {
          self.write_bit(false)?; /* Without using temporal prediction */
        }
        self.write_bit(segmentation.update_data)?;
      }
      if segmentation.update_data {
        for i in 0..8 {
          for j in 0..SegLvl::SEG_LVL_MAX as usize {
            self.write_bit(segmentation.features[i][j])?;
            if segmentation.features[i][j] {
              let bits = seg_feature_bits[j];
              let data = segmentation.data[i][j];
              if seg_feature_is_signed[j] {
                self.write_signed(bits + 1, data)?;
              } else {
                self.write(bits, data)?;
              }
            }
          }
        }
      }
    }
    Ok(())
  }

  fn write_delta_q(&mut self, delta_q: i8) -> io::Result<()> {
    self.write_bit(delta_q != 0)?;
    if delta_q != 0 {
      self.write_signed(6 + 1, delta_q)?;
    }
    Ok(())
  }
}
