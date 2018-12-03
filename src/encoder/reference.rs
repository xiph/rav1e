use std::rc::Rc;

use context::CDFContext;
use encoder::deblock::DeblockState;
use encoder::frame::Frame;
use partition::REF_FRAMES;
use plane::Plane;

#[derive(Debug, Clone)]
pub struct ReferenceFrame {
  pub order_hint: u32,
  pub frame: Frame,
  pub input_hres: Plane,
  pub input_qres: Plane,
  pub cdfs: CDFContext
}

#[derive(Debug, Clone)]
pub struct ReferenceFramesSet {
  pub frames: [Option<Rc<ReferenceFrame>>; (REF_FRAMES as usize)],
  pub deblock: [DeblockState; (REF_FRAMES as usize)]
}

impl ReferenceFramesSet {
  pub fn new() -> ReferenceFramesSet {
    ReferenceFramesSet {
      frames: Default::default(),
      deblock: Default::default()
    }
  }
}

#[allow(dead_code, non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReferenceMode {
  SINGLE = 0,
  COMPOUND = 1,
  SELECT = 2
}

pub const ALL_REF_FRAMES_MASK: u32 = (1 << REF_FRAMES) - 1;
