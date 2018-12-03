use std::io;

use bitstream_io::BigEndian;
use bitstream_io::BitWriter;

use context::seg_feature_bits;
use context::seg_feature_is_signed;
use context::SegLvl;
use context::PLANES;
use ec::BCodeWriter;
use encoder::frame::FrameInvariants;
use encoder::frame::FrameState;
use encoder::frame::FrameType;
use encoder::reference::ReferenceMode;
use encoder::reference::ALL_REF_FRAMES_MASK;
use encoder::sequence::Sequence;
use encoder::ChromaSampling;
use encoder::OBU_Type;
use encoder::PRIMARY_REF_BITS;
use encoder::PRIMARY_REF_NONE;
use lrf::RESTORE_NONE;
use partition::GlobalMVMode;
use partition::ALTREF_FRAME;
use partition::INTER_REFS_PER_FRAME;
use partition::LAST_FRAME;
use partition::REF_FRAMES;
use partition::REF_FRAMES_LOG2;

pub trait UncompressedHeader {
  // Start of OBU Headers
  fn write_obu_header(
    &mut self, obu_type: OBU_Type, obu_extension: u32
  ) -> io::Result<()>;
  fn write_sequence_header_obu(
    &mut self, seq: &mut Sequence, fi: &FrameInvariants
  ) -> io::Result<()>;
  fn write_frame_header_obu(
    &mut self, seq: &Sequence, fi: &FrameInvariants, fs: &FrameState
  ) -> io::Result<()>;
  fn write_sequence_header(
    &mut self, seq: &mut Sequence, fi: &FrameInvariants
  ) -> io::Result<()>;
  fn write_color_config(&mut self, seq: &mut Sequence) -> io::Result<()>;
  // End of OBU Headers

  fn write_frame_size(&mut self, fi: &FrameInvariants) -> io::Result<()>;
  fn write_deblock_filter_a(
    &mut self, fi: &FrameInvariants, fs: &FrameState
  ) -> io::Result<()>;
  fn write_deblock_filter_b(
    &mut self, fi: &FrameInvariants, fs: &FrameState
  ) -> io::Result<()>;
  fn write_frame_cdef(
    &mut self, seq: &Sequence, fi: &FrameInvariants
  ) -> io::Result<()>;
  fn write_frame_lrf(
    &mut self, seq: &Sequence, fi: &FrameInvariants
  ) -> io::Result<()>;
  fn write_segment_data(
    &mut self, fi: &FrameInvariants, fs: &FrameState
  ) -> io::Result<()>;
}
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

impl<W: io::Write> UncompressedHeader for BitWriter<W, BigEndian> {
  // Start of OBU Headers
  // Write OBU Header syntax
  fn write_obu_header(
    &mut self, obu_type: OBU_Type, obu_extension: u32
  ) -> io::Result<()> {
    self.write_bit(false)?; // forbidden bit.
    self.write(4, obu_type as u32)?;
    self.write_bit(obu_extension != 0)?;
    self.write_bit(true)?; // obu_has_payload_length_field
    self.write_bit(false)?; // reserved

    if obu_extension != 0 {
      assert!(false);
      //self.write(8, obu_extension & 0xFF)?; size += 8;
    }

    Ok(())
  }

  fn write_sequence_header_obu(
    &mut self, seq: &mut Sequence, fi: &FrameInvariants
  ) -> io::Result<()> {
    self.write(3, seq.profile)?; // profile, 3 bits
    self.write(1, 0)?; // still_picture
    self.write(1, 0)?; // reduced_still_picture
    self.write_bit(false)?; // display model present
    self.write_bit(false)?; // no timing info present
    self.write(5, 0)?; // one operating point
    self.write(12, 0)?; // idc
    self.write(5, 31)?; // level
    self.write(1, 0)?; // tier
    if seq.reduced_still_picture_hdr {
      assert!(false);
    }

    self.write_sequence_header(seq, fi)?;

    self.write_color_config(seq)?;

    self.write_bit(seq.film_grain_params_present)?;

    self.write_bit(true)?; // add_trailing_bits

    Ok(())
  }

  #[allow(unused)]
  fn write_frame_header_obu(
    &mut self, seq: &Sequence, fi: &FrameInvariants, fs: &FrameState
  ) -> io::Result<()> {
    if seq.reduced_still_picture_hdr {
      assert!(fi.show_existing_frame);
      assert!(fi.frame_type == FrameType::KEY);
      assert!(fi.show_frame);
    } else {
      if fi.show_existing_frame {
        self.write_bit(true)?; // show_existing_frame=1
        self.write(3, fi.frame_to_show_map_idx)?;

        //TODO:
        /* temporal_point_info();
          if seq.decoder_model_info_present_flag &&
            timing_info.equal_picture_interval == 0 {
          // write frame_presentation_delay;
        }
        if seq.frame_id_numbers_present_flag {
          // write display_frame_id;
        }*/

        self.write_bit(true)?; // trailing bit
        self.byte_align()?;
        return Ok(());
      }
      self.write_bit(false)?; // show_existing_frame=0
      self.write(2, fi.frame_type as u32)?;
      self.write_bit(fi.show_frame)?; // show frame

      if fi.show_frame {
        //TODO:
        /* temporal_point_info();
          if seq.decoder_model_info_present_flag &&
          timing_info.equal_picture_interval == 0 {
        // write frame_presentation_delay;*/
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

    if seq.force_screen_content_tools == 2 {
      self.write_bit(fi.allow_screen_content_tools != 0)?;
    } else {
      assert!(fi.allow_screen_content_tools == seq.force_screen_content_tools);
    }

    if fi.allow_screen_content_tools == 2 {
      if seq.force_integer_mv == 2 {
        self.write_bit(fi.force_integer_mv != 0)?;
      } else {
        assert!(fi.force_integer_mv == seq.force_integer_mv);
      }
    } else {
      assert!(fi.allow_screen_content_tools == seq.force_screen_content_tools);
    }

    if seq.frame_id_numbers_present_flag {
      assert!(false); // Not supported by rav1e yet!
                      //TODO:
                      //let frame_id_len = seq.frame_id_length;
                      //self.write(frame_id_len, fi.current_frame_id);
    }

    let mut frame_size_override_flag = false;
    if fi.frame_type == FrameType::SWITCH {
      frame_size_override_flag = true;
    } else if seq.reduced_still_picture_hdr {
      frame_size_override_flag = false;
    } else {
      self.write_bit(frame_size_override_flag)?; // frame size overhead flag
    }

    if seq.enable_order_hint {
      let n = seq.order_hint_bits_minus_1 + 1;
      let mask = (1 << n) - 1;
      self.write(n, fi.order_hint & mask)?;
    }

    if fi.error_resilient || fi.intra_only {
    } else {
      self.write(PRIMARY_REF_BITS, fi.primary_ref_frame)?;
    }

    if seq.decoder_model_info_present_flag {
      assert!(false); // Not supported by rav1e yet!
    }

    if fi.frame_type == FrameType::KEY {
      if !fi.show_frame {
        // unshown keyframe (forward keyframe)
        assert!(false); // Not supported by rav1e yet!
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
      if (fi.error_resilient && seq.enable_order_hint) {
        assert!(false); // Not supported by rav1e yet!
                        //for _ in 0..REF_FRAMES {
                        //  self.write(order_hint_bits_minus_1,ref_order_hint[i])?; // order_hint
                        //}
      }
    }

    // if KEY or INTRA_ONLY frame
    // FIXME: Not sure whether putting frame/render size here is good idea
    if fi.intra_only {
      if frame_size_override_flag {
        assert!(false); // Not supported by rav1e yet!
      }
      if seq.enable_superres {
        assert!(false); // Not supported by rav1e yet!
      }
      self.write_bit(false)?; // render_and_frame_size_different
                              //if render_and_frame_size_different { }
      if fi.allow_screen_content_tools != 0 && true
      /* UpscaledWidth == FrameWidth */
      {
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
        if seq.enable_order_hint {
          self.write_bit(frame_refs_short_signaling)?;
          if frame_refs_short_signaling {
            assert!(false); // Not supported by rav1e yet!
          }
        }

        for i in 0..INTER_REFS_PER_FRAME {
          if !frame_refs_short_signaling {
            self.write(REF_FRAMES_LOG2 as u32, fi.ref_frames[i] as u8)?;
          }
          if seq.frame_id_numbers_present_flag {
            assert!(false); // Not supported by rav1e yet!
          }
        }
        if fi.error_resilient && frame_size_override_flag {
          assert!(false); // Not supported by rav1e yet!
        } else {
          if frame_size_override_flag {
            assert!(false); // Not supported by rav1e yet!
          }
          if seq.enable_superres {
            assert!(false); // Not supported by rav1e yet!
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
        if fi.error_resilient || !seq.enable_ref_frame_mvs {
        } else {
          self.write_bit(fi.use_ref_frame_mvs)?;
        }
      }
    }

    if !seq.reduced_still_picture_hdr && !fi.disable_cdf_update {
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
    self.write_bit(false)?; // y dc delta q
    self.write_bit(false)?; // uv dc delta q
    self.write_bit(false)?; // uv ac delta q
    self.write_bit(false)?; // no qm

    // segmentation
    self.write_segment_data(fi, fs)?;

    // delta_q
    self.write_bit(false)?; // delta_q_present_flag: no delta q

    // delta_lf_params in the spec
    self.write_deblock_filter_a(fi, fs)?;

    // code for features not yet implemented....

    // loop_filter_params in the spec
    self.write_deblock_filter_b(fi, fs)?;

    // cdef
    self.write_frame_cdef(seq, fi)?;

    // loop restoration
    self.write_frame_lrf(seq, fi)?;

    self.write_bit(false)?; // tx mode == TX_MODE_SELECT ?

    let mut reference_select = false;
    if !fi.intra_only {
      reference_select = fi.reference_mode != ReferenceMode::SINGLE;
      self.write_bit(reference_select)?;
    }

    let skip_mode_allowed = seq.get_skip_mode_allowed(fi, reference_select);
    if skip_mode_allowed {
      self.write_bit(false)?; // skip_mode_present
    }

    if fi.intra_only || fi.error_resilient || !seq.enable_warped_motion {
    } else {
      self.write_bit(fi.allow_warped_motion)?; // allow_warped_motion
    }

    self.write_bit(fi.use_reduced_tx_set)?; // reduced tx

    // global motion
    if !fi.intra_only {
      for i in LAST_FRAME..ALTREF_FRAME + 1 {
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

    if seq.film_grain_params_present && fi.show_frame {
      unimplemented!();
    }

    if fi.large_scale_tile {
      unimplemented!();
    }
    self.write_bit(true)?; // trailing bit
    self.byte_align()?;

    Ok(())
  }

  fn write_sequence_header(
    &mut self, seq: &mut Sequence, fi: &FrameInvariants
  ) -> io::Result<()> {
    self.write_frame_size(fi)?;

    if !seq.reduced_still_picture_hdr {
      seq.frame_id_numbers_present_flag = false;
      seq.frame_id_length = FRAME_ID_LENGTH as u32;
      seq.delta_frame_id_length = DELTA_FRAME_ID_LENGTH as u32;

      self.write_bit(seq.frame_id_numbers_present_flag)?;

      if seq.frame_id_numbers_present_flag {
        // We must always have delta_frame_id_length < frame_id_length,
        // in order for a frame to be referenced with a unique delta.
        // Avoid wasting bits by using a coding that enforces this restriction.
        self.write(4, seq.delta_frame_id_length - 2)?;
        self.write(3, seq.frame_id_length - seq.delta_frame_id_length - 1)?;
      }
    }

    self.write_bit(seq.use_128x128_superblock)?;
    self.write_bit(seq.enable_filter_intra)?;
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

    self.write_bit(high_bd)?; // high bit depth

    if seq.bit_depth == 12 {
      self.write_bit(true)?; // 12-bit
    }

    if seq.profile != 1 {
      self.write_bit(seq.monochrome)?; // monochrome?
    } else {
      unimplemented!(); // 4:4:4 sampling at 8 or 10 bits
    }

    self.write_bit(false)?; // No color description present

    if seq.monochrome {
      assert!(false);
    }

    self.write_bit(false)?; // color range

    let subsampling_x = seq.chroma_sampling != ChromaSampling::Cs444;
    let subsampling_y = seq.chroma_sampling == ChromaSampling::Cs420;

    if seq.bit_depth == 12 {
      self.write_bit(subsampling_x)?;

      if subsampling_x {
        self.write_bit(subsampling_y)?;
      }
    }

    if !subsampling_y {
      unimplemented!(); // 4:2:2 or 4:4:4 sampling
    }

    self.write(2, 0)?; // chroma_sample_position == CSP_UNKNOWN

    self.write_bit(false)?; // separate uv delta q

    Ok(())
  }
  // End of OBU Headers

  fn write_frame_size(&mut self, fi: &FrameInvariants) -> io::Result<()> {
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

  fn write_deblock_filter_a(
    &mut self, fi: &FrameInvariants, fs: &FrameState
  ) -> io::Result<()> {
    if fi.delta_q_present {
      if !fi.allow_intrabc {
        self.write_bit(fs.deblock.block_deltas_enabled)?;
      }
      if fs.deblock.block_deltas_enabled {
        self.write(2, fs.deblock.block_delta_shift)?;
        self.write_bit(fs.deblock.block_delta_multi)?;
      }
    }
    Ok(())
  }

  fn write_deblock_filter_b(
    &mut self, fi: &FrameInvariants, fs: &FrameState
  ) -> io::Result<()> {
    assert!(fs.deblock.levels[0] < 64);
    self.write(6, fs.deblock.levels[0])?; // loop deblocking filter level 0
    assert!(fs.deblock.levels[1] < 64);
    self.write(6, fs.deblock.levels[1])?; // loop deblocking filter level 1
    if PLANES > 1 && (fs.deblock.levels[0] > 0 || fs.deblock.levels[1] > 0) {
      assert!(fs.deblock.levels[2] < 64);
      self.write(6, fs.deblock.levels[2])?; // loop deblocking filter level 2
      assert!(fs.deblock.levels[3] < 64);
      self.write(6, fs.deblock.levels[3])?; // loop deblocking filter level 3
    }
    self.write(3, fs.deblock.sharpness)?; // deblocking filter sharpness
    self.write_bit(fs.deblock.deltas_enabled)?; // loop deblocking filter deltas enabled
    if fs.deblock.deltas_enabled {
      self.write_bit(fs.deblock.delta_updates_enabled)?; // deltas updates enabled
      if fs.deblock.delta_updates_enabled {
        // conditionally write ref delta updates
        let prev_ref_deltas = if fi.primary_ref_frame == PRIMARY_REF_NONE {
          [1, 0, 0, 0, 0, -1, -1, -1]
        } else {
          fi.rec_buffer.deblock
            [fi.ref_frames[fi.primary_ref_frame as usize] as usize]
            .ref_deltas
        };
        for i in 0..REF_FRAMES {
          let update = fs.deblock.ref_deltas[i] != prev_ref_deltas[i];
          self.write_bit(update)?;
          if update {
            self.write_signed(7, fs.deblock.ref_deltas[i])?;
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
          let update = fs.deblock.mode_deltas[i] != prev_mode_deltas[i];
          self.write_bit(update)?;
          if update {
            self.write_signed(7, fs.deblock.mode_deltas[i])?;
          }
        }
      }
    }
    Ok(())
  }

  fn write_frame_cdef(
    &mut self, seq: &Sequence, fi: &FrameInvariants
  ) -> io::Result<()> {
    if seq.enable_cdef {
      assert!(fi.cdef_damping >= 3);
      assert!(fi.cdef_damping <= 6);
      self.write(2, fi.cdef_damping - 3)?;
      assert!(fi.cdef_bits < 4);
      self.write(2, fi.cdef_bits)?; // cdef bits
      for i in 0..(1 << fi.cdef_bits) {
        assert!(fi.cdef_y_strengths[i] < 64);
        assert!(fi.cdef_uv_strengths[i] < 64);
        self.write(6, fi.cdef_y_strengths[i])?; // cdef y strength
        self.write(6, fi.cdef_uv_strengths[i])?; // cdef uv strength
      }
    }
    Ok(())
  }

  fn write_frame_lrf(
    &mut self, seq: &Sequence, fi: &FrameInvariants
  ) -> io::Result<()> {
    if seq.enable_restoration && !fi.allow_intrabc {
      // && !self.lossless
      let mut use_lrf = false;
      let mut use_chroma_lrf = false;
      for i in 0..PLANES {
        self.write(2, fi.lrf_types[i])?; // filter type by plane
        if fi.lrf_types[i] != RESTORE_NONE {
          use_lrf = true;
          if i > 0 {
            use_chroma_lrf = true;
          }
        }
      }
      if use_lrf {
        // At present, we're locked to a restoration unit size equal to superblock size.
        // Signal as such.
        if seq.use_128x128_superblock {
          self.write(1, 0)?; // do not double the restoration unit from 128x128
        } else {
          self.write(1, 0)?; // do not double the restoration unit from 64x64
        }

        if use_chroma_lrf {
          // until we're able to support restoration units larger than
          // the chroma superblock size, we can't perform LRF for
          // anything other than 4:4:4 and 4:2:0
          assert!(
            seq.chroma_sampling == ChromaSampling::Cs444
              || seq.chroma_sampling == ChromaSampling::Cs420
          );
          if seq.chroma_sampling == ChromaSampling::Cs420 {
            self.write(1, 1)?; // halve the chroma restoration unit in both directions
          }
        }
      }
    }
    Ok(())
  }

  fn write_segment_data(
    &mut self, fi: &FrameInvariants, fs: &FrameState
  ) -> io::Result<()> {
    self.write_bit(fs.segmentation.enabled)?;
    if fs.segmentation.enabled {
      if fi.primary_ref_frame == PRIMARY_REF_NONE {
        assert_eq!(fs.segmentation.update_map, true);
        assert_eq!(fs.segmentation.update_data, true);
      } else {
        self.write_bit(fs.segmentation.update_map)?;
        if fs.segmentation.update_map {
          self.write_bit(false)?; /* Without using temporal prediction */
        }
        self.write_bit(fs.segmentation.update_data)?;
      }
      if fs.segmentation.update_data {
        for i in 0..8 {
          for j in 0..SegLvl::SEG_LVL_MAX as usize {
            self.write_bit(fs.segmentation.features[i][j])?;
            if fs.segmentation.features[i][j] {
              let bits = seg_feature_bits[j];
              let data = fs.segmentation.data[i][j];
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
}
