use context::SegLvl;

#[derive(Copy, Clone, Debug)]
pub struct SegmentationState {
  pub enabled: bool,
  pub update_data: bool,
  pub update_map: bool,
  pub preskip: bool,
  pub last_active_segid: u8,
  pub features: [[bool; SegLvl::SEG_LVL_MAX as usize]; 8],
  pub data: [[i16; SegLvl::SEG_LVL_MAX as usize]; 8]
}

impl Default for SegmentationState {
  fn default() -> Self {
    SegmentationState {
      enabled: false,
      update_data: false,
      update_map: false,
      preskip: true,
      last_active_segid: 0,
      features: [[false; SegLvl::SEG_LVL_MAX as usize]; 8],
      data: [[0; SegLvl::SEG_LVL_MAX as usize]; 8]
    }
  }
}
