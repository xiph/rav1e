use context::PLANES;
use partition::REF_FRAMES;

#[derive(Copy, Clone, Debug)]
pub struct DeblockState {
  pub levels: [u8; PLANES + 1], // Y vertical edges, Y horizontal, U, V
  pub sharpness: u8,
  pub deltas_enabled: bool,
  pub delta_updates_enabled: bool,
  pub ref_deltas: [i8; REF_FRAMES],
  pub mode_deltas: [i8; 2],
  pub block_deltas_enabled: bool,
  pub block_delta_shift: u8,
  pub block_delta_multi: bool
}

impl Default for DeblockState {
  fn default() -> Self {
    DeblockState {
      levels: [8, 8, 4, 4],
      sharpness: 0,
      deltas_enabled: false, // requires delta_q_enabled
      delta_updates_enabled: false,
      ref_deltas: [1, 0, 0, 0, 0, -1, -1, -1],
      mode_deltas: [0, 0],
      block_deltas_enabled: false,
      block_delta_shift: 0,
      block_delta_multi: false
    }
  }
}
