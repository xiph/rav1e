use crate::api::internal::InterConfig;
use crate::config::EncoderConfig;
use crate::context::{BlockOffset, FrameBlocks, TileBlockOffset};
use crate::cpu_features::CpuFeatureLevel;
use crate::dist::get_satd;
use crate::encoder::{
  FrameInvariants, FrameState, Sequence, IMPORTANCE_BLOCK_SIZE,
};
use crate::frame::{AsRegion, PlaneOffset};
use crate::me::{estimate_tile_motion, FrameMEStats, RefMEStats};
use crate::partition::{get_intra_edges, BlockSize};
use crate::predict::{pred_cfl_ac, IntraParam, PredictionMode};
use crate::tiling::{Area, PlaneRegion, TileRect};
use crate::Pixel;
use rayon::iter::*;
use rust_hawktracer::*;
use std::sync::Arc;
use v_frame::frame::Frame;
use v_frame::pixel::CastFromPrimitive;
use v_frame::plane::Plane;

pub(crate) const IMP_BLOCK_MV_UNITS_PER_PIXEL: i64 = 8;
pub(crate) const IMP_BLOCK_SIZE_IN_MV_UNITS: i64 =
  IMPORTANCE_BLOCK_SIZE as i64 * IMP_BLOCK_MV_UNITS_PER_PIXEL;
pub(crate) const IMP_BLOCK_AREA_IN_MV_UNITS: i64 =
  IMP_BLOCK_SIZE_IN_MV_UNITS * IMP_BLOCK_SIZE_IN_MV_UNITS;

#[hawktracer(estimate_intra_costs)]
pub(crate) fn estimate_intra_costs<T: Pixel>(
  temp_frame: &mut Frame<T>, frame: &Frame<T>, bit_depth: usize,
  planes: usize, cpu_feature_level: CpuFeatureLevel,
) -> Box<[u32]> {
  let h_in_imp_b = frame.planes[0].cfg.height / IMPORTANCE_BLOCK_SIZE;
  let w_in_imp_b = frame.planes[0].cfg.width / IMPORTANCE_BLOCK_SIZE;
  let mut intra_costs = Vec::with_capacity(h_in_imp_b * w_in_imp_b);

  estimate_plane_intra_costs::<T, 0, 0>(
    &mut temp_frame.planes[0],
    &frame.planes[0],
    None,
    bit_depth,
    cpu_feature_level,
    &mut intra_costs,
    w_in_imp_b,
    h_in_imp_b,
  );

  for (plane_rec, plane) in
    temp_frame.planes[1..planes].iter_mut().zip(&frame.planes[1..planes])
  {
    let xdec = plane.cfg.xdec;
    let ydec = plane.cfg.ydec;
    (match (xdec, ydec) {
      (0, 0) => estimate_plane_intra_costs::<T, 0, 0>,
      (_, 0) => estimate_plane_intra_costs::<T, 1, 0>,
      (_, _) => estimate_plane_intra_costs::<T, 1, 1>,
    })(
      plane_rec,
      plane,
      Some(&frame.planes[0]),
      bit_depth,
      cpu_feature_level,
      &mut intra_costs,
      w_in_imp_b,
      h_in_imp_b,
    );
  }

  intra_costs.into_boxed_slice()
}

fn estimate_plane_intra_costs<
  T: Pixel,
  const XDEC: usize,
  const YDEC: usize,
>(
  plane_rec: &mut Plane<T>, plane: &Plane<T>, plane_y: Option<&Plane<T>>,
  bit_depth: usize, cpu_feature_level: CpuFeatureLevel,
  intra_costs: &mut Vec<u32>, w_in_imp_b: usize, h_in_imp_b: usize,
) {
  let bsize = BlockSize::from_width_and_height(
    IMPORTANCE_BLOCK_SIZE >> XDEC,
    IMPORTANCE_BLOCK_SIZE >> YDEC,
  );
  let tx_size = bsize.tx_size();
  let width = IMPORTANCE_BLOCK_SIZE >> XDEC;
  let height = IMPORTANCE_BLOCK_SIZE >> YDEC;
  let mut ac = [0i16; IMPORTANCE_BLOCK_SIZE * IMPORTANCE_BLOCK_SIZE];
  for y_ in 0..h_in_imp_b {
    for x_ in 0..w_in_imp_b {
      let x = (x_ * IMPORTANCE_BLOCK_SIZE) as isize >> XDEC;
      let y = (y_ * IMPORTANCE_BLOCK_SIZE) as isize >> YDEC;
      let plane_org = plane.region(Area::Rect { x, y, width, height });

      // TODO: other intra prediction modes.
      let edge_buf = get_intra_edges(
        &plane.as_region(),
        TileBlockOffset(BlockOffset { x: x_, y: y_ }),
        0,
        0,
        bsize,
        PlaneOffset { x, y },
        tx_size,
        bit_depth,
        Some(PredictionMode::DC_PRED),
        false,
        IntraParam::None,
      );

      let mut plane_after_prediction_region =
        plane_rec.region_mut(Area::Rect { x, y, width, height });

      PredictionMode::DC_PRED.predict_intra(
        TileRect { x: x as usize, y: y as usize, width, height },
        &mut plane_after_prediction_region,
        tx_size,
        bit_depth,
        &[], // Not used by DC_PRED
        IntraParam::None,
        None, // Not used by DC_PRED
        &edge_buf,
        cpu_feature_level,
      );

      let plane_after_prediction_region =
        plane_rec.region(Area::Rect { x, y, width, height });

      let mut satd = get_satd(
        &plane_org,
        &plane_after_prediction_region,
        bsize.width(),
        bsize.height(),
        bit_depth,
        cpu_feature_level,
      );

      if let Some(plane) = plane_y {
        let luma = &plane.region(Area::Rect {
          x: x << XDEC,
          y: y << YDEC,
          width: width << XDEC,
          height: height << YDEC,
        });
        pred_cfl_ac::<T, XDEC, YDEC>(
          &mut ac,
          luma,
          bsize,
          0,
          0,
          cpu_feature_level,
        );

        let thresh = 4;
        let mut progress = 1;
        for alpha in 1..=15 {
          if alpha > thresh && progress < alpha {
            break;
          }
          let mut flag = 0;
          for sign in (-1..1).step_by(2) {
            let mut plane_after_prediction_region =
              plane_rec.region_mut(Area::Rect { x, y, width, height });

            PredictionMode::UV_CFL_PRED.predict_intra(
              TileRect { x: x as usize, y: y as usize, width, height },
              &mut plane_after_prediction_region,
              tx_size,
              bit_depth,
              &ac,
              IntraParam::Alpha(sign * alpha),
              None,
              &edge_buf,
              cpu_feature_level,
            );

            let plane_after_prediction_region =
              plane_rec.region(Area::Rect { x, y, width, height });

            let new_satd = get_satd(
              &plane_org,
              &plane_after_prediction_region,
              bsize.width(),
              bsize.height(),
              bit_depth,
              cpu_feature_level,
            );
            if new_satd <= satd {
              satd = new_satd;
              flag = thresh;
            }
          }
          progress += flag;
        }
      }

      let intra_cost = satd << (XDEC + YDEC);

      if intra_costs.len() < h_in_imp_b * w_in_imp_b {
        intra_costs.push(intra_cost);
      } else {
        intra_costs[y_ * w_in_imp_b + x_] += intra_cost;
      }
    }
  }
}

#[hawktracer(estimate_importance_block_difference)]
pub(crate) fn estimate_importance_block_difference<T: Pixel>(
  frame: Arc<Frame<T>>, ref_frame: Arc<Frame<T>>,
) -> f64 {
  let plane_org = &frame.planes[0];
  let plane_ref = &ref_frame.planes[0];
  let h_in_imp_b = plane_org.cfg.height / IMPORTANCE_BLOCK_SIZE;
  let w_in_imp_b = plane_org.cfg.width / IMPORTANCE_BLOCK_SIZE;

  let mut imp_block_costs = 0;

  (0..h_in_imp_b).for_each(|y| {
    (0..w_in_imp_b).for_each(|x| {
      // Coordinates of the top-left corner of the reference block, in MV
      // units.
      let region_org = plane_org.region(Area::Rect {
        x: (x * IMPORTANCE_BLOCK_SIZE) as isize,
        y: (y * IMPORTANCE_BLOCK_SIZE) as isize,
        width: IMPORTANCE_BLOCK_SIZE,
        height: IMPORTANCE_BLOCK_SIZE,
      });

      let region_ref = plane_ref.region(Area::Rect {
        x: (x * IMPORTANCE_BLOCK_SIZE) as isize,
        y: (y * IMPORTANCE_BLOCK_SIZE) as isize,
        width: IMPORTANCE_BLOCK_SIZE,
        height: IMPORTANCE_BLOCK_SIZE,
      });

      let sum_8x8_block = |region: &PlaneRegion<T>| {
        region
          .rows_iter()
          .map(|row| {
            // 16-bit precision is sufficient for an 8px row, as IMPORTANCE_BLOCK_SIZE * (2^12 - 1) < 2^16 - 1,
            // so overflow is not possible
            row.iter().map(|pixel| u16::cast_from(*pixel)).sum::<u16>() as i64
          })
          .sum::<i64>()
      };

      let histogram_org_sum = sum_8x8_block(&region_org);
      let histogram_ref_sum = sum_8x8_block(&region_ref);

      let count = (IMPORTANCE_BLOCK_SIZE * IMPORTANCE_BLOCK_SIZE) as i64;

      let mean = (((histogram_org_sum + count / 2) / count)
        - ((histogram_ref_sum + count / 2) / count))
        .abs();

      imp_block_costs += mean as u64;
    });
  });

  imp_block_costs as f64 / (w_in_imp_b * h_in_imp_b) as f64
}

#[hawktracer(estimate_inter_costs)]
pub(crate) fn estimate_inter_costs<T: Pixel>(
  frame: Arc<Frame<T>>, ref_frame: Arc<Frame<T>>, bit_depth: usize,
  mut config: EncoderConfig, sequence: Arc<Sequence>, buffer: RefMEStats,
  planes: usize,
) -> f64 {
  config.low_latency = true;
  config.speed_settings.multiref = false;
  let inter_cfg = InterConfig::new(&config);
  let last_fi = FrameInvariants::new_key_frame(
    Arc::new(config),
    sequence,
    0,
    Box::new([]),
  );
  let mut fi = FrameInvariants::new_inter_frame(
    &last_fi,
    &inter_cfg,
    0,
    1,
    2,
    false,
    Box::new([]),
  )
  .unwrap();

  // Compute the motion vectors.
  let mut fs = FrameState::new_with_frame_and_me_stats_and_rec(
    &fi,
    Arc::clone(&frame),
    buffer,
    // We do not use this field, so we can avoid the expensive allocation
    Arc::new(Frame {
      planes: [
        Plane::new(0, 0, 0, 0, 0, 0),
        Plane::new(0, 0, 0, 0, 0, 0),
        Plane::new(0, 0, 0, 0, 0, 0),
      ],
    }),
  );
  compute_motion_vectors(&mut fi, &mut fs, &inter_cfg);

  // Estimate inter costs
  let h_in_imp_b = frame.planes[0].cfg.height / IMPORTANCE_BLOCK_SIZE;
  let w_in_imp_b = frame.planes[0].cfg.width / IMPORTANCE_BLOCK_SIZE;
  let stats = &fs.frame_me_stats.read().expect("poisoned lock")[0];

  let mut inter_costs = 0;
  for (plane_org, plane_ref) in
    frame.planes.iter().zip(&ref_frame.planes).take(planes)
  {
    let xdec = plane_org.cfg.xdec;
    let ydec = plane_org.cfg.ydec;
    inter_costs += (match (xdec, ydec) {
      (0, 0) => estimate_plane_inter_costs::<T, 0, 0>,
      (_, 0) => estimate_plane_inter_costs::<T, 1, 0>,
      (_, _) => estimate_plane_inter_costs::<T, 1, 1>,
    })(
      plane_org, plane_ref, bit_depth, &fi, stats, w_in_imp_b, h_in_imp_b,
    );
  }
  inter_costs as f64 / (w_in_imp_b * h_in_imp_b) as f64
}

fn estimate_plane_inter_costs<
  T: Pixel,
  const XDEC: usize,
  const YDEC: usize,
>(
  plane_org: &Plane<T>, plane_ref: &Plane<T>, bit_depth: usize,
  fi: &FrameInvariants<T>, stats: &FrameMEStats, w_in_imp_b: usize,
  h_in_imp_b: usize,
) -> u64 {
  let bsize = BlockSize::from_width_and_height(
    IMPORTANCE_BLOCK_SIZE >> XDEC,
    IMPORTANCE_BLOCK_SIZE >> YDEC,
  );
  let mut inter_costs = 0;
  let width = IMPORTANCE_BLOCK_SIZE >> XDEC;
  let height = IMPORTANCE_BLOCK_SIZE >> YDEC;
  (0..h_in_imp_b).for_each(|y| {
    (0..w_in_imp_b).for_each(|x| {
      let mv = stats[y * 2][x * 2].mv;

      // Coordinates of the top-left corner of the reference block, in MV
      // units.
      let reference_x = x as i64 * IMP_BLOCK_SIZE_IN_MV_UNITS + mv.col as i64;
      let reference_y = y as i64 * IMP_BLOCK_SIZE_IN_MV_UNITS + mv.row as i64;

      let region_org = plane_org.region(Area::Rect {
        x: (x * width) as isize,
        y: (y * height) as isize,
        width,
        height,
      });

      let region_ref = plane_ref.region(Area::Rect {
        x: (reference_x as isize >> XDEC)
          / IMP_BLOCK_MV_UNITS_PER_PIXEL as isize,
        y: (reference_y as isize >> YDEC)
          / IMP_BLOCK_MV_UNITS_PER_PIXEL as isize,
        width,
        height,
      });

      inter_costs += (get_satd(
        &region_org,
        &region_ref,
        bsize.width(),
        bsize.height(),
        bit_depth,
        fi.cpu_feature_level,
      ) as u64)
        << (XDEC + YDEC);
    });
  });
  inter_costs
}

#[hawktracer(compute_motion_vectors)]
pub(crate) fn compute_motion_vectors<T: Pixel>(
  fi: &mut FrameInvariants<T>, fs: &mut FrameState<T>, inter_cfg: &InterConfig,
) {
  let mut blocks = FrameBlocks::new(fi.w_in_b, fi.h_in_b);
  fi.sequence
    .tiling
    .tile_iter_mut(fs, &mut blocks)
    .collect::<Vec<_>>()
    .into_par_iter()
    .for_each(|mut ctx| {
      let ts = &mut ctx.ts;
      estimate_tile_motion(fi, ts, inter_cfg);
    });
}
