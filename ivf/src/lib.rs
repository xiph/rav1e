/// Simple ivf muxer
///

use bitstream_io::{BitWriter, LittleEndian};
use std::io;

pub fn write_ivf_header(
  output_file: &mut dyn io::Write, width: usize, height: usize,
  framerate_num: usize,
  framerate_den: usize
) {
  let mut bw = BitWriter::endian(output_file, LittleEndian);
  bw.write_bytes(b"DKIF").unwrap();
  bw.write(16, 0).unwrap(); // version
  bw.write(16, 32).unwrap(); // version
  bw.write_bytes(b"AV01").unwrap();
  bw.write(16, width as u16).unwrap();
  bw.write(16, height as u16).unwrap();
  bw.write(32, framerate_num as u32).unwrap();
  bw.write(32, framerate_den as u32).unwrap();
  bw.write(32, 0).unwrap();
  bw.write(32, 0).unwrap();
}

pub fn write_ivf_frame(
  output_file: &mut dyn io::Write, pts: u64, data: &[u8]
) {
  let mut bw = BitWriter::endian(output_file, LittleEndian);
  bw.write(32, data.len() as u32).unwrap();
  bw.write(64, pts).unwrap();
  bw.write_bytes(data).unwrap();
}
