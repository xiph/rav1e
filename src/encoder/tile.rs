use bitstream_io::BigEndian;
use bitstream_io::BitWriter;

use cdef::cdef_filter_frame;
use context::BlockContext;
use context::CDFContext;
use context::ContextWriter;
use context::RestorationContext;
use context::SuperBlockOffset;
use deblock::deblock_filter_frame;
use deblock::deblock_filter_optimize;
use ec::WriterEncoder;
use ec::WriterRecorder;
use encoder::frame::FrameInvariants;
use encoder::frame::FrameState;
use encoder::partition::encode_partition_bottomup;
use encoder::partition::encode_partition_topdown;
use encoder::sequence::Sequence;
use encoder::PRIMARY_REF_NONE;
use me::estimate_motion_ss2;
use me::estimate_motion_ss4;
use partition::BlockSize;
use partition::MotionVector;
use partition::INTER_REFS_PER_FRAME;
use partition::REF_FRAMES;
use rdo::rdo_cdef_decision;

pub fn encode_tile(
  sequence: &mut Sequence, fi: &FrameInvariants, fs: &mut FrameState,
  bit_depth: usize
) -> Vec<u8> {
  let mut w = WriterEncoder::new();

  let fc = if fi.primary_ref_frame == PRIMARY_REF_NONE {
    CDFContext::new(fi.base_q_idx)
  } else {
    match fi.rec_buffer.frames
      [fi.ref_frames[fi.primary_ref_frame as usize] as usize]
    {
      Some(ref rec) => rec.cdfs,
      None => CDFContext::new(fi.base_q_idx)
    }
  };

  let bc = BlockContext::new(fi.w_in_b, fi.h_in_b);
  // For now, restoration unit size is locked to superblock size.
  let rc = RestorationContext::new(fi.sb_width, fi.sb_height);
  let mut cw = ContextWriter::new(fc, bc, rc);

  // initial coarse ME loop
  let mut frame_pmvs = Vec::new();

  for sby in 0..fi.sb_height {
    for sbx in 0..fi.sb_width {
      let sbo = SuperBlockOffset { x: sbx, y: sby };
      let bo = sbo.block_offset(0, 0);
      let mut pmvs: [Option<MotionVector>; REF_FRAMES] = [None; REF_FRAMES];
      for i in 0..INTER_REFS_PER_FRAME {
        let r = fi.ref_frames[i] as usize;
        if pmvs[r].is_none() {
          assert!(!sequence.use_128x128_superblock);
          pmvs[r] = estimate_motion_ss4(
            fi,
            fs,
            BlockSize::BLOCK_64X64,
            r,
            &bo,
            sequence.bit_depth
          );
        }
      }
      frame_pmvs.push(pmvs);
    }
  }

  // main loop
  for sby in 0..fi.sb_height {
    cw.bc.reset_left_contexts();

    for sbx in 0..fi.sb_width {
      let mut w_pre_cdef = WriterRecorder::new();
      let mut w_post_cdef = WriterRecorder::new();
      let mut cdef_index = 0;
      let sbo = SuperBlockOffset { x: sbx, y: sby };
      let bo = sbo.block_offset(0, 0);
      cw.bc.cdef_coded = false;
      cw.bc.code_deltas = fi.delta_q_present;

      // Do subsampled ME
      let mut pmvs: [[Option<MotionVector>; REF_FRAMES]; 5] =
        [[None; REF_FRAMES]; 5];
      for i in 0..INTER_REFS_PER_FRAME {
        let r = fi.ref_frames[i] as usize;
        if pmvs[0][r].is_none() {
          pmvs[0][r] = frame_pmvs[sby * fi.sb_width + sbx][r];
          if let Some(pmv) = pmvs[0][r] {
            let pmv_w = if sbx > 0 {
              frame_pmvs[sby * fi.sb_width + sbx - 1][r]
            } else {
              None
            };
            let pmv_e = if sbx < fi.sb_width - 1 {
              frame_pmvs[sby * fi.sb_width + sbx + 1][r]
            } else {
              None
            };
            let pmv_n = if sby > 0 {
              frame_pmvs[sby * fi.sb_width + sbx - fi.sb_width][r]
            } else {
              None
            };
            let pmv_s = if sby < fi.sb_height - 1 {
              frame_pmvs[sby * fi.sb_width + sbx + fi.sb_width][r]
            } else {
              None
            };

            assert!(!sequence.use_128x128_superblock);
            pmvs[1][r] = estimate_motion_ss2(
              fi,
              fs,
              BlockSize::BLOCK_32X32,
              r,
              &sbo.block_offset(0, 0),
              &[Some(pmv), pmv_w, pmv_n],
              sequence.bit_depth
            );
            pmvs[2][r] = estimate_motion_ss2(
              fi,
              fs,
              BlockSize::BLOCK_32X32,
              r,
              &sbo.block_offset(8, 0),
              &[Some(pmv), pmv_e, pmv_n],
              sequence.bit_depth
            );
            pmvs[3][r] = estimate_motion_ss2(
              fi,
              fs,
              BlockSize::BLOCK_32X32,
              r,
              &sbo.block_offset(0, 8),
              &[Some(pmv), pmv_w, pmv_s],
              sequence.bit_depth
            );
            pmvs[4][r] = estimate_motion_ss2(
              fi,
              fs,
              BlockSize::BLOCK_32X32,
              r,
              &sbo.block_offset(8, 8),
              &[Some(pmv), pmv_e, pmv_s],
              sequence.bit_depth
            );
          }
        }
      }

      // Encode SuperBlock
      if fi.config.speed_settings.encode_bottomup {
        encode_partition_bottomup(
          sequence,
          fi,
          fs,
          &mut cw,
          &mut w_pre_cdef,
          &mut w_post_cdef,
          BlockSize::BLOCK_64X64,
          &bo,
          &pmvs
        );
      } else {
        encode_partition_topdown(
          sequence,
          fi,
          fs,
          &mut cw,
          &mut w_pre_cdef,
          &mut w_post_cdef,
          BlockSize::BLOCK_64X64,
          &bo,
          &None,
          &pmvs
        );
      }

      // CDEF has to be decisded before loop restoration, but coded after
      if cw.bc.cdef_coded {
        cdef_index = rdo_cdef_decision(&sbo, fi, fs, &mut cw, bit_depth);
        cw.bc.set_cdef(&sbo, cdef_index);
      }

      // loop restoration must be decided last but coded before anything else
      if sequence.enable_restoration {}

      // Once loop restoration is coded, we can replay the initial block bits
      w_pre_cdef.replay(&mut w);

      if cw.bc.cdef_coded {
        // CDEF index must be written in the middle, we can code it now
        cw.write_cdef(&mut w, cdef_index, fi.cdef_bits);
        // ...and then finally code what comes after the CDEF index
        w_post_cdef.replay(&mut w);
      }
    }
  }
  /* TODO: Don't apply if lossless */
  deblock_filter_optimize(fi, fs, &mut cw.bc, bit_depth);
  if fs.deblock.levels[0] != 0 || fs.deblock.levels[1] != 0 {
    deblock_filter_frame(fs, &mut cw.bc, bit_depth);
  }
  /* TODO: Don't apply if lossless */
  if sequence.enable_cdef {
    cdef_filter_frame(fi, &mut fs.rec, &mut cw.bc, bit_depth);
  }

  fs.cdfs = cw.fc;
  fs.cdfs.reset_counts();

  let mut h = w.done();
  h.push(0); // superframe anti emulation
  h
}

#[allow(unused)]
pub fn write_tile_group_header(
  tile_start_and_end_present_flag: bool
) -> Vec<u8> {
  let mut buf = Vec::new();
  {
    let mut bw = BitWriter::endian(&mut buf, BigEndian);
    bw.write_bit(tile_start_and_end_present_flag).unwrap();
    bw.byte_align().unwrap();
  }
  buf.clone()
}
