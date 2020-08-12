use std::{
    collections::HashMap,
    fs, mem,
    path::{Path, PathBuf},
};

use clap::{App, Arg};
use ndarray::{Array, Axis};
use rayon::prelude::*;
use serde::Serialize;

/// Per-plane data, accumulated every frame.
struct PlaneData {
    /// Whether the frame is an intra frame.
    is_intra: bool,
    /// DC qidx chosen for this frame (assumed to be constant throughout the frame).
    dc_qidx: u8,
    /// AC qidx chosen for this frame (assumed to be constant throughout the frame).
    ac_qidx: u8,
    /// DC values.
    dcs: Vec<f64>,
    /// Sum of squares of AC values (M<sub>2,n</sub> from [1], assuming mean = 0).
    ///
    /// [1]: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    ac_m: f64,
    /// Count of AC values.
    ac_count: u64,
    /// DC factor, computed as N × σ<sup>2</sup> (eq. 20 from [av1-metric.pdf]).
    ///
    /// [av1-metric.pdf]: http://dgql.org/~derf/av1-metric.pdf
    dc_factor: f64,
    /// AC factor, computed as N × σ<sup>2</sup> (eq. 20 from [av1-metric.pdf]).
    ///
    /// [av1-metric.pdf]: http://dgql.org/~derf/av1-metric.pdf
    ac_factor: f64,
}

/// Quantizer weights for every analyzed frame, indexed by quantizer index.
#[derive(Serialize)]
struct PerFramePlaneData {
    /// Weights of the DC quantizers across all 6 quantizers.
    dc_weights: Vec<Vec<f64>>,
    /// Weights of the AC quantizers across all 6 quantizers.
    ac_weights: Vec<Vec<f64>>,
    /// Weights of the DC quantizers across 2 of this plane's quantizers.
    plane_dc_weights: Vec<Vec<f64>>,
    /// Weights of the AC quantizers across 2 of this plane's quantizers.
    plane_ac_weights: Vec<Vec<f64>>,
}

impl Default for PerFramePlaneData {
    fn default() -> Self {
        Self {
            dc_weights: vec![vec![]; 256],
            ac_weights: vec![vec![]; 256],
            plane_dc_weights: vec![vec![]; 256],
            plane_ac_weights: vec![vec![]; 256],
        }
    }
}

fn main() {
    let m = App::new("process-vstats")
        .about("Analyses data dumps from the luma_chroma_collect1 rav1e branch")
        .arg(
            Arg::with_name("data-dir")
                .short('d')
                .long("data-dir")
                .takes_value(true)
                .required(true)
                .about("Path to the rav1e data dump folder")
                .long_about(
                    "This folder must have files called \"{sequence}-q{q}.log\" \
                    where {sequence} is given with --sequence and {q} is in 1..=239.",
                ),
        )
        .arg(
            Arg::with_name("sequence")
                .short('s')
                .long("sequence")
                .takes_value(true)
                .multiple_values(true)
                .min_values(1)
                .required(true)
                .about("Sequence to analyze"),
        )
        .arg(
            Arg::with_name("output-dir")
                .short('o')
                .long("output-dir")
                .takes_value(true)
                .required(true)
                .about("Path to the output folder"),
        )
        .get_matches();

    let data_dir: PathBuf = m.value_of_t("data-dir").unwrap();
    let output_dir: PathBuf = m.value_of_t("output-dir").unwrap();

    let sequences: Vec<_> = m.values_of("sequence").unwrap().collect();
    sequences
        .par_iter()
        .for_each(|sequence| process(&data_dir, &output_dir, sequence));
}

fn process(data_dir: &Path, output_dir: &Path, sequence: &str) {
    println!("analyzing {}", sequence);

    let mut per_frame_data = HashMap::<bool, HashMap<u8, PerFramePlaneData>>::new();

    for q in 1..=239 {
        let path = data_dir.join(&format!("{}-q{}.log", sequence, q));
        if !path.exists() {
            continue;
        }

        let contents = fs::read_to_string(path).unwrap();

        let mut frame_data = [None::<PlaneData>, None::<PlaneData>, None::<PlaneData>];

        for line in contents.lines() {
            let mut split = line.split_ascii_whitespace();
            let type_ = split.next().unwrap();
            let mut split = split.map(|val| val.parse::<i64>().unwrap());

            match type_ {
                "f" => {
                    let mut weight_sums = [0.; 3];

                    for plane in 0..3 {
                        if let Some(plane_data) = &mut frame_data[plane] {
                            let dcs = Array::from(mem::replace(&mut plane_data.dcs, vec![]));
                            let dc_n = dcs.len() as f64;
                            let dc_s = dcs.var_axis(Axis(0), 0.).into_scalar();
                            plane_data.dc_factor = dc_n * dc_s;

                            let ac_n = plane_data.ac_count as f64;
                            let ac_s = plane_data.ac_m / ac_n;
                            plane_data.ac_factor = ac_n * ac_s;

                            weight_sums[plane] += plane_data.dc_factor;
                            weight_sums[plane] += plane_data.ac_factor;
                        }
                    }

                    let weight_sum = weight_sums.iter().sum::<f64>();

                    for plane in 0..3 {
                        if let Some(plane_data) = frame_data[plane].take() {
                            let per_frame_plane_data = per_frame_data
                                .entry(plane_data.is_intra)
                                .or_default()
                                .entry(plane as u8)
                                .or_default();

                            let dc_weight = plane_data.dc_factor / weight_sum;
                            per_frame_plane_data.dc_weights[plane_data.dc_qidx as usize]
                                .push(dc_weight);
                            let ac_weight = plane_data.ac_factor / weight_sum;
                            per_frame_plane_data.ac_weights[plane_data.ac_qidx as usize]
                                .push(ac_weight);

                            let plane_dc_weight = plane_data.dc_factor / weight_sums[plane];
                            per_frame_plane_data.plane_dc_weights[plane_data.dc_qidx as usize]
                                .push(plane_dc_weight);
                            let plane_ac_weight = plane_data.ac_factor / weight_sums[plane];
                            per_frame_plane_data.plane_ac_weights[plane_data.ac_qidx as usize]
                                .push(plane_ac_weight);
                        }
                    }
                }
                "b" => {
                    let s1 = split.next().unwrap();
                    let s2 = split.next().unwrap();
                    let plane = split.next().unwrap();
                    let pixel_count = split.next().unwrap() as u64;
                    let dc_qidx = split.next().unwrap();
                    let ac_qidx = split.next().unwrap();
                    let _block_is_intra = split.next().unwrap();
                    let frame_is_intra = split.next().unwrap();

                    let dc = s1 as f64 / pixel_count as f64;
                    let ac_m = s2 as f64 - (s1 * s1) as f64 / pixel_count as f64;

                    let plane_data = &mut frame_data[plane as usize];
                    if plane_data.is_none() {
                        *plane_data = Some(PlaneData {
                            is_intra: frame_is_intra != 0,
                            dc_qidx: dc_qidx as _,
                            ac_qidx: ac_qidx as _,
                            dcs: vec![],
                            ac_m: 0.,
                            ac_count: 0,
                            dc_factor: 0.,
                            ac_factor: 0.,
                        });
                    }

                    let plane_data = plane_data.as_mut().unwrap();
                    plane_data.dcs.push(dc);
                    plane_data.ac_m += ac_m;
                    plane_data.ac_count += pixel_count - 1;

                    assert_eq!(plane_data.dc_qidx, dc_qidx as u8);
                    assert_eq!(plane_data.ac_qidx, ac_qidx as u8);
                }
                _ => panic!("unexpected type {}", type_),
            }
        }
    }

    let mut per_frame_data_str = HashMap::<&str, HashMap<&str, PerFramePlaneData>>::new();
    for (is_intra, is_intra_str) in &[(true, "true"), (false, "false")] {
        let per_frame_data = per_frame_data.get_mut(is_intra).unwrap();
        let per_frame_data_str = per_frame_data_str.entry(is_intra_str).or_default();
        for (plane, plane_str) in &[(0, "0"), (1, "1"), (2, "2")] {
            per_frame_data_str.insert(plane_str, per_frame_data.remove(plane).unwrap());
        }
    }

    let json = serde_json::to_vec(&per_frame_data_str).unwrap();
    fs::write(output_dir.join(&format!("{}.json", sequence)), &json).unwrap();
}
