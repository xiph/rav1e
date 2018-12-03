use api::FrameInfo;
use encoder::frame::FrameInvariants;
use encoder::ChromaSampling;
use encoder::MAX_NUM_OPERATING_POINTS;
use partition::INTER_REFS_PER_FRAME;

#[derive(Copy, Clone)]
pub struct Sequence {
  // OBU Sequence header of AV1
  pub profile: u8,
  pub num_bits_width: u32,
  pub num_bits_height: u32,
  pub bit_depth: usize,
  pub chroma_sampling: ChromaSampling,
  pub max_frame_width: u32,
  pub max_frame_height: u32,
  pub frame_id_numbers_present_flag: bool,
  pub frame_id_length: u32,
  pub delta_frame_id_length: u32,
  pub use_128x128_superblock: bool,
  pub order_hint_bits_minus_1: u32,
  pub force_screen_content_tools: u32, // 0 - force off
  // 1 - force on
  // 2 - adaptive
  pub force_integer_mv: u32, // 0 - Not to force. MV can be in 1/4 or 1/8
  // 1 - force to integer
  // 2 - adaptive
  pub still_picture: bool, // Video is a single frame still picture
  pub reduced_still_picture_hdr: bool, // Use reduced header for still picture
  pub monochrome: bool,    // Monochrome video
  pub enable_filter_intra: bool, // enables/disables filterintra
  pub enable_intra_edge_filter: bool, // enables/disables corner/edge/upsampling
  pub enable_interintra_compound: bool, // enables/disables interintra_compound
  pub enable_masked_compound: bool,   // enables/disables masked compound
  pub enable_dual_filter: bool,       // 0 - disable dual interpolation filter
  // 1 - enable vert/horiz filter selection
  pub enable_order_hint: bool, // 0 - disable order hint, and related tools
  // jnt_comp, ref_frame_mvs, frame_sign_bias
  // if 0, enable_jnt_comp and
  // enable_ref_frame_mvs must be set zs 0.
  pub enable_jnt_comp: bool, // 0 - disable joint compound modes
  // 1 - enable it
  pub enable_ref_frame_mvs: bool, // 0 - disable ref frame mvs
  // 1 - enable it
  pub enable_warped_motion: bool, // 0 - disable warped motion for sequence
  // 1 - enable it for the sequence
  pub enable_superres: bool, // 0 - Disable superres for the sequence, and disable
  //     transmitting per-frame superres enabled flag.
  // 1 - Enable superres for the sequence, and also
  //     enable per-frame flag to denote if superres is
  //     enabled for that frame.
  pub enable_cdef: bool,        // To turn on/off CDEF
  pub enable_restoration: bool, // To turn on/off loop restoration
  pub operating_points_cnt_minus_1: usize,
  pub operating_point_idc: [u16; MAX_NUM_OPERATING_POINTS],
  pub display_model_info_present_flag: bool,
  pub decoder_model_info_present_flag: bool,
  pub level: [[usize; 2]; MAX_NUM_OPERATING_POINTS], // minor, major
  pub tier: [usize; MAX_NUM_OPERATING_POINTS], // seq_tier in the spec. One bit: 0
  // or 1.
  pub film_grain_params_present: bool,
  pub separate_uv_delta_q: bool
}

impl Sequence {
  pub fn new(info: &FrameInfo) -> Sequence {
    let width_bits = 32 - (info.width as u32).leading_zeros();
    let height_bits = 32 - (info.height as u32).leading_zeros();
    assert!(width_bits <= 16);
    assert!(height_bits <= 16);

    let profile = if info.bit_depth == 12 {
      2
    } else if info.chroma_sampling == ChromaSampling::Cs444 {
      1
    } else {
      0
    };

    let mut operating_point_idc = [0 as u16; MAX_NUM_OPERATING_POINTS];
    let mut level = [[1, 2 as usize]; MAX_NUM_OPERATING_POINTS];
    let mut tier = [0 as usize; MAX_NUM_OPERATING_POINTS];

    for i in 0..MAX_NUM_OPERATING_POINTS {
      operating_point_idc[i] = 0;
      level[i][0] = 1; // minor
      level[i][1] = 2; // major
      tier[i] = 0;
    }

    Sequence {
      profile,
      num_bits_width: width_bits,
      num_bits_height: height_bits,
      bit_depth: info.bit_depth,
      chroma_sampling: info.chroma_sampling,
      max_frame_width: info.width as u32,
      max_frame_height: info.height as u32,
      frame_id_numbers_present_flag: false,
      frame_id_length: 0,
      delta_frame_id_length: 0,
      use_128x128_superblock: false,
      order_hint_bits_minus_1: 5,
      force_screen_content_tools: 0,
      force_integer_mv: 2,
      still_picture: false,
      reduced_still_picture_hdr: false,
      monochrome: false,
      enable_filter_intra: true,
      enable_intra_edge_filter: true,
      enable_interintra_compound: false,
      enable_masked_compound: false,
      enable_dual_filter: false,
      enable_order_hint: true,
      enable_jnt_comp: false,
      enable_ref_frame_mvs: false,
      enable_warped_motion: false,
      enable_superres: false,
      enable_cdef: true,
      enable_restoration: true,
      operating_points_cnt_minus_1: 0,
      operating_point_idc,
      display_model_info_present_flag: false,
      decoder_model_info_present_flag: false,
      level,
      tier,
      film_grain_params_present: false,
      separate_uv_delta_q: false
    }
  }

  pub fn get_relative_dist(&self, a: u32, b: u32) -> i32 {
    let diff = a as i32 - b as i32;
    let m = 1 << self.order_hint_bits_minus_1;
    (diff & (m - 1)) - (diff & m)
  }

  pub fn get_skip_mode_allowed(
    &self, fi: &FrameInvariants, reference_select: bool
  ) -> bool {
    if fi.intra_only || !reference_select || !self.enable_order_hint {
      false
    } else {
      let mut forward_idx: isize = -1;
      let mut backward_idx: isize = -1;
      let mut forward_hint = 0;
      let mut backward_hint = 0;
      for i in 0..INTER_REFS_PER_FRAME {
        if let Some(ref rec) = fi.rec_buffer.frames[fi.ref_frames[i] as usize]
        {
          let ref_hint = rec.order_hint;
          if self.get_relative_dist(ref_hint, fi.order_hint) < 0 {
            if forward_idx < 0
              || self.get_relative_dist(ref_hint, forward_hint) > 0
            {
              forward_idx = i as isize;
              forward_hint = ref_hint;
            }
          } else if self.get_relative_dist(ref_hint, fi.order_hint) > 0 {
            if backward_idx < 0
              || self.get_relative_dist(ref_hint, backward_hint) > 0
            {
              backward_idx = i as isize;
              backward_hint = ref_hint;
            }
          }
        }
      }
      if forward_idx < 0 {
        false
      } else if backward_idx >= 0 {
        // set skip_mode_frame
        true
      } else {
        let mut second_forward_idx: isize = -1;
        let mut second_forward_hint = 0;
        for i in 0..INTER_REFS_PER_FRAME {
          if let Some(ref rec) =
            fi.rec_buffer.frames[fi.ref_frames[i] as usize]
          {
            let ref_hint = rec.order_hint;
            if self.get_relative_dist(ref_hint, forward_hint) < 0 {
              if second_forward_idx < 0
                || self.get_relative_dist(ref_hint, second_forward_hint) > 0
              {
                second_forward_idx = i as isize;
                second_forward_hint = ref_hint;
              }
            }
          }
        }
        if second_forward_idx < 0 {
          false
        } else {
          // set skip_mode_frame
          true
        }
      }
    }
  }
}
