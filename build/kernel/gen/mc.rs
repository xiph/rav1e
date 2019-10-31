use std::env;
use std::fs::File;
use std::io::{BufWriter, Cursor, Write};
use std::path::Path;

use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct EightTapFrac(bool, bool);
impl EightTapFrac {
  fn fracs() -> &'static [Self] {
    const C: &'static [EightTapFrac] = &[
      EightTapFrac(true, true),
      EightTapFrac(true, false),
      EightTapFrac(false, true),
      EightTapFrac(false, false),
    ];
    C
  }
  fn fn_suffix(&self) -> &'static str {
    match self {
      EightTapFrac(true, true) => "0_0",
      EightTapFrac(true, false) => "0_y",
      EightTapFrac(false, true) => "x_0",
      EightTapFrac(false, false) => "x_y",
    }
  }
  fn module_name(&self) -> Ident {
    let name = format!("f_{}", self.fn_suffix());
    Ident::new(&name, Span::call_site())
  }
  /// In the (false, false) case, I've diverged slightly from the
  /// reference implementation. In the first loop over the rows, I've
  /// swapped the write to intermediates so that we write to the
  /// corresponding transposed position. This allows the second row
  /// loop to load the row from intermediates with a stride of 1,
  /// saving us from having to use a vector gather.
  /// Ditto for `Self::prep_body`.
  fn put_body(&self, px: PixelType, b: Block) -> TokenStream {
    let width = b.w();
    let height = b.h();

    let calc_ty = PrimType::I32;
    let px = match px {
      PixelType::U8 => PrimType::U8,
      PixelType::U16 => PrimType::U16,
    };
    // the simd width:
    let step = 8usize;
    let calc_simd = quote! { Simd::<[#calc_ty; #step]> };

    let mut idxs = TokenStream::default();
    for i in 0..step {
      idxs.extend(quote! {
        #i,
      });
    }
    let idxs = quote!(Simd::<[usize; #step]>::new(#idxs));

    let x_filter = Ident::new("x_filter", Span::call_site());
    let y_filter = Ident::new("y_filter", Span::call_site());

    let t_ident = |j| Ident::new(&format!("t{}", j), Span::call_site());
    let load_unit_stride_src = |px, src, j| {
      let t_ident = t_ident(j);
      let src = if j != 0 { quote!(#src.add(#j)) } else { src };
      let t = quote! {
        Simd::<[#px; #step]>::from_slice_unaligned_unchecked({
          from_raw_parts(#src, #step)
        })
      };
      (quote!(let #t_ident = #calc_simd ::from_cast(#t);), quote!(#t_ident))
    };
    let load_src = |px, src, j, stride| {
      let t_ident = t_ident(j);

      let src = if j != 0 { quote!(#src.add(#j)) } else { src };
      let t = quote! {{
        let ptrs = Simd::<[*const #px; #step]>::splat(#src);
        let idxs = #idxs * #stride;
        ptrs.add(idxs)
          .read(Simd::<[m8; #step]>::splat(true),
                Default::default())
      }};
      (quote!(let #t_ident = #calc_simd ::from_cast(#t);), quote!(#t_ident))
    };

    let run_filter = |src, filter, bits: &[_]| {
      let mut t = quote! {
        (#filter * #src).wrapping_sum() as i32
      };
      for bits in bits.iter() {
        t = quote!(round_shift(#t, #bits));
      }

      t
    };

    let outer_loop = |out: &mut TokenStream,
                      inner_loop: &dyn Fn(&mut TokenStream),
                      div: Option<usize>| {
      let div = div.unwrap_or(1);
      if width * height / div <= MAX_UNROLL {
        out.extend(quote! {
          // ensure #inner doesn't mess up the start of the next row:
          let dst_start = dst;
          let src_start = src;
        });
        for i in 0..height {
          if i != 0 {
            out.extend(quote! {
              dst = dst_start.add(dst_stride as usize);
              src = src_start.add(src_stride as usize);
              // ensure #inner doesn't mess up the start of the next row:
              let dst_start = dst;
              let src_start = src;
            });
          }
          inner_loop(out);
        }
      } else {
        let mut inner = TokenStream::default();
        inner_loop(&mut inner);
        out.extend(quote! {
          for _ in 0..#height {
            // ensure #inner doesn't mess up the start of the next row:
            let dst_start = dst;
            let src_start = src;

            #inner

            dst = dst_start.add(dst_stride as usize);
            src = src_start.add(src_stride as usize);
          }
        });
      }
    };

    let mut out = TokenStream::default();
    if px == PrimType::U8 {
      out.extend(quote! {
        #[cfg(debug_assertions)]
        {
          if bit_depth != 8 {
            panic!("put_8tap kernel expects 8bit depth on u8 pixels");
          }
        }
      });
    }
    out.extend(quote! {
      let max_sample_val = ((1 << bit_depth) - 1) as i32;
      let intermediate_bits = 4 - if bit_depth == 12 { 2 } else { 0 };
    });

    match self {
      EightTapFrac(true, true) => {
        let step = b.w().min(px.avx2_width());
        let inner_loop = |out: &mut TokenStream| {
          for j in (0..b.w()).step_by(step) {
            let src = if j != 0 { quote!(src.add(#j)) } else { quote!(src) };
            let dst = if j != 0 { quote!(dst.add(#j)) } else { quote!(dst) };
            let t = quote! {
              let t = Simd::<[#px; #step]>::from_slice_unaligned_unchecked({
                from_raw_parts(#src, #step)
              });
              t.write_to_slice_unaligned_unchecked({
                from_raw_parts_mut(#dst, #step)
              });
            };
            out.extend(t);
          }
        };
        // unroll both loops for small blocks
        outer_loop(&mut out, &inner_loop, Some(px.avx2_width() * 8));
      }
      EightTapFrac(true, false) => {
        let stride = quote!((src_stride as usize));
        out.extend(quote! {
          //let src = src.go_up(3);
          src = src.offset(-3 * (src_stride as isize));
          let y_filter = #calc_simd ::from_cast(y_filter);
        });
        let inner_loop = |out: &mut TokenStream| {
          let mut ts = Vec::new();
          for j in 0..b.w() {
            let (load, t) = load_src(px, quote!(src), j, &stride);
            ts.push((t, j));
            out.extend(load);
          }
          for (t, j) in ts.into_iter() {
            let t = run_filter(t, &y_filter, &[quote!(7)]);
            let mut t = quote!(#t.max(0));
            let dst = if j != 0 { quote!(dst.add(#j)) } else { quote!(dst) };
            if px == PrimType::U16 {
              t = quote!(#t.min(max_sample_val as _));
            }
            out.extend(quote! {
              *#dst = #t as #px;
            });
          }
        };
        outer_loop(&mut out, &inner_loop, None);
      }
      EightTapFrac(false, true) => {
        out.extend(quote! {
          //let src = src.go_left(3);
          src = src.offset(-3);
          let x_filter = #calc_simd ::from_cast(x_filter);
        });
        let bits = &[quote!(7 - intermediate_bits), quote!(intermediate_bits)];
        let inner_loop = |out: &mut TokenStream| {
          let mut ts = Vec::new();
          for j in 0..b.w() {
            let (load, t) = load_unit_stride_src(px, quote!(src), j);
            ts.push((t, j));
            out.extend(load);
          }
          for (t, j) in ts.into_iter() {
            let t = run_filter(t, &x_filter, bits);
            let mut t = quote!(#t.max(0));
            let dst = if j != 0 { quote!(dst.add(#j)) } else { quote!(dst) };
            if px == PrimType::U16 {
              t = quote!(#t.min(max_sample_val as _));
            }
            out.extend(quote! {
              *#dst = #t as #px;
            });
          }
        };
        outer_loop(&mut out, &inner_loop, None);
      }
      EightTapFrac(false, false) => {
        // TODO: completely unroll 4xN blocks. These
        // are the only cases where 0..8.min(width) => 0..4
        // Since 4xN blocks are small anyway, might as well
        // just unroll them completely to avoid that weird
        // cg + #c < #width if block below.
        let mut inner1 = TokenStream::default();

        let bits1 = &[quote!(7 - intermediate_bits)];
        let bits2 = &[quote!(7 + intermediate_bits)];

        for c in 0..8.min(width) {
          inner1.extend(quote! {
            let t = from_raw_parts(src.add(#c + cg), #step);
            let t = Simd::<[#px; #step]>::from_slice_unaligned_unchecked(t);
            let t = #calc_simd ::from_cast(t);
          });

          let t = run_filter(quote!(t), &x_filter, bits1);
          let write = quote! {
            *iptr.add(#c * (#height + 7usize)) = #t as i16;
          };
          if 8 > width {
            inner1.extend(quote! {
              if cg + #c < #width {
                #write
              }
            });
          } else {
            inner1.extend(write);
          }
        }

        let inner1 = inner1;

        let mut inner2 = TokenStream::default();

        for c in 0..8.min(width) {
          let ty = PrimType::I16;
          inner2.extend(quote! {
            let t = from_raw_parts(iptr.add(#c * (#height + 7usize)), #step);
            let t = Simd::<[#ty; #step]>::from_slice_unaligned_unchecked(t);
            let t = #calc_simd ::from_cast(t);
          });

          let t = run_filter(quote!(t), &y_filter, bits2);
          let mut t = quote!(#t.max(0));
          if px == PrimType::U16 {
            t = quote!(#t.min(max_sample_val as _));
          }
          let write = quote! {
            *dst.add(cg + #c) = #t as #px;
          };
          if 8 > width {
            inner2.extend(quote! {
              if cg + #c < #width {
                #write
              }
            });
          } else {
            inner2.extend(write);
          }
        }
        out.extend(quote! {
          src = src.offset(-3isize * (src_stride as isize) - 3);
          let x_filter = #calc_simd ::from_cast(x_filter);
          let y_filter = #calc_simd ::from_cast(y_filter);
          let mut intermediate: AlignedArray<[i16; 8 * (#height + 7)]> =
            AlignedArray::uninitialized();
          for cg in (0..#width).step_by(8) {
            let mut iptr = intermediate.as_mut_ptr();
            let mut src = src;
            for r in 0..#height + 7 {
              #inner1
              iptr = iptr.add(1);
              src = src.add(src_stride as usize);
            }
            let mut iptr = intermediate.as_ptr();
            let mut dst = dst;
            for r in 0..#height {
              #inner2
              iptr = iptr.add(1);
              dst = dst.add(dst_stride as usize);
            }
          }
        });
      }
    }

    out
  }
  fn prep_body(&self, px: PixelType, b: Block) -> TokenStream {
    let width = b.w();
    let height = b.h();

    let calc_ty = PrimType::I32;
    let px = match px {
      PixelType::U8 => PrimType::U8,
      PixelType::U16 => PrimType::U16,
    };
    // the simd width:
    let step = 8usize;
    let calc_simd = quote! { Simd::<[#calc_ty; #step]> };

    let x_filter = Ident::new("x_filter", Span::call_site());
    let y_filter = Ident::new("y_filter", Span::call_site());

    let t_ident = |j| Ident::new(&format!("t{}", j), Span::call_site());
    let load_unit_stride_src = |px, src, j| {
      let t_ident = t_ident(j);
      let src = if j != 0 { quote!(#src.add(#j)) } else { src };
      let t = quote! {
        Simd::<[#px; #step]>::from_slice_unaligned_unchecked({
          from_raw_parts(#src, #step)
        })
      };
      (quote!(let #t_ident = #calc_simd ::from_cast(#t);), quote!(#t_ident))
    };
    let load_src = |px, src, j, stride| {
      let t_ident = t_ident(j);
      let mut idxs = TokenStream::default();
      for i in 0..step {
        idxs.extend(quote! {
          #i,
        });
      }
      let idxs = quote! {
        Simd::<[usize; #step]>::new(#idxs) * #stride
      };
      let src = if j != 0 { quote!(#src.add(#j)) } else { src };
      let t = quote! {{
        let ptrs = Simd::<[*const #px; #step]>::splat(#src);
        ptrs.add(#idxs)
          .read(Simd::<[m8; #step]>::splat(true),
                Default::default())
      }};
      (quote!(let #t_ident = #calc_simd ::from_cast(#t);), quote!(#t_ident))
    };

    let run_filter = |src, filter, bits: &[_]| {
      let mut t = quote!((#filter * #src).wrapping_sum() as i32);
      for bits in bits.iter() {
        t = quote!(round_shift(#t, #bits));
      }

      t
    };

    let outer_loop = |out: &mut TokenStream,
                      inner_loop: &dyn Fn(&mut TokenStream),
                      div: Option<usize>| {
      let div = div.unwrap_or(1);
      if width * height / div <= MAX_UNROLL {
        out.extend(quote! {
          // ensure #inner doesn't mess up the start of the next row:
          let tmp_start = tmp;
          let src_start = src;
        });
        for i in 0..height {
          if i != 0 {
            out.extend(quote! {
              tmp = tmp.add(#width);
              src = src_start.add(src_stride as usize);
              // ensure #inner doesn't mess up the start of the next row:
              let tmp_start = tmp;
              let src_start = src;
            });
          }
          inner_loop(out);
        }
      } else {
        let mut inner = TokenStream::default();
        inner_loop(&mut inner);
        out.extend(quote! {
          for _ in 0..#height {
            // ensure #inner doesn't mess up the start of the next row:
            let tmp_start = tmp;
            let src_start = src;

            #inner

            tmp = tmp_start.add(#width);
            src = src_start.add(src_stride as usize);
          }
        });
      }
    };

    let bits = &[quote!(7 - intermediate_bits)];

    let mut out = TokenStream::default();
    out.extend(quote! {
      let max_sample_val = ((1 << bit_depth) - 1) as i32;
      let intermediate_bits = 4 - if bit_depth == 12 { 2 } else { 0 };
    });

    match self {
      EightTapFrac(true, true) => {
        let step = b.w().min(px.avx2_width());
        let inner_loop = |out: &mut TokenStream| {
          for j in (0..b.w()).step_by(step) {
            let src = if j != 0 { quote!(src.add(#j)) } else { quote!(src) };
            let dst = if j != 0 { quote!(tmp.add(#j)) } else { quote!(tmp) };
            let t = quote! {
              let t = Simd::<[#px; #step]>::from_slice_unaligned_unchecked({
                from_raw_parts(#src, #step)
              });
              let t = Simd::<[i16; #step]>::from_cast(t) << (intermediate_bits as u32);
              t.write_to_slice_unaligned_unchecked({
                from_raw_parts_mut(#dst, #step)
              });
            };
            out.extend(t);
          }
        };
        outer_loop(&mut out, &inner_loop, Some(px.avx2_width()));
      }
      EightTapFrac(true, false) => {
        let stride = quote!((src_stride as usize));
        out.extend(quote! {
          // src.go_up(3)
          src = src.offset(-3isize * (src_stride as isize));
          let y_filter = #calc_simd ::from_cast(y_filter);
        });
        let inner_loop = |out: &mut TokenStream| {
          let mut ts = Vec::new();
          for j in 0..b.w() {
            let (load, t) = load_src(px, quote!(src), j, &stride);
            ts.push((t, j));
            out.extend(load);
          }
          for (t, j) in ts.into_iter() {
            let t = run_filter(t, &y_filter, bits);
            out.extend(quote! {
              *tmp.add(#j) = #t as i16;
            });
          }
        };
        outer_loop(&mut out, &inner_loop, None);
      }
      EightTapFrac(false, true) => {
        out.extend(quote! {
          // src.go_left(3)
          src = src.offset(-3isize);
          let x_filter = #calc_simd ::from_cast(x_filter);
        });
        let inner_loop = |out: &mut TokenStream| {
          let mut ts = Vec::new();
          for j in 0..b.w() {
            let (load, t) = load_unit_stride_src(px, quote!(src), j);
            ts.push((t, j));
            out.extend(load);
          }
          for (t, j) in ts.into_iter() {
            let t = run_filter(t, &x_filter, bits);
            out.extend(quote! {
              *tmp.add(#j) = #t as i16;
            });
          }
        };
        outer_loop(&mut out, &inner_loop, None);
      }
      EightTapFrac(false, false) => {
        // TODO? this is never completely unrolled
        // TODO: completely unroll 4xN blocks. These
        // are the only cases where 0..8.min(width) => 0..4
        // Since 4xN blocks are small anyway, might as well
        // just unroll them completely to avoid that weird
        // cg + #c < #width if block below.

        let mut inner1 = TokenStream::default();

        for c in 0..8.min(width) {
          inner1.extend(quote! {
            let t = from_raw_parts(src.add(#c + cg), #step);
            let t = Simd::<[#px; #step]>::from_slice_unaligned_unchecked(t);
            let t = #calc_simd ::from_cast(t);
          });
          let t = run_filter(quote!(t), &x_filter, bits);
          let write = quote! {
            *iptr.add(#c * (#height + 7usize)) = #t as i16;
          };
          if 8 > width {
            inner1.extend(quote! {
              if cg + #c < #width {
                #write
              }
            });
          } else {
            inner1.extend(write);
          }
        }
        let inner1 = inner1;

        let mut inner2 = TokenStream::default();

        let bits = &[quote!(7)];
        for c in 0..8.min(width) {
          let ty = PrimType::I16;
          inner2.extend(quote! {
            let t = from_raw_parts(iptr.add(#c * (#height + 7usize)), #step);
            let t = Simd::<[#ty; #step]>::from_slice_unaligned_unchecked(t);
            let t = #calc_simd ::from_cast(t);
          });
          let t = run_filter(quote!(t), &y_filter, bits);
          let write = quote! {
            *tmp.add(cg + #c) = #t as i16;
          };
          if 8 > width {
            inner2.extend(quote! {
              if cg + #c < #width {
                #write
              }
            });
          } else {
            inner2.extend(write);
          }
        }
        out.extend(quote! {
          src = src.offset(-3isize * (src_stride as isize) - 3);
          let x_filter = #calc_simd ::from_cast(x_filter);
          let y_filter = #calc_simd ::from_cast(y_filter);
          let mut intermediate: AlignedArray<[i16; 8 * (#height + 7)]> =
            AlignedArray::uninitialized();
          for cg in (0..#width).step_by(8) {
            let mut iptr = intermediate.as_mut_ptr();
            let mut src = src;
            for r in 0..#height + 7 {
              #inner1
              iptr = iptr.add(1);
              src = src.add(src_stride as usize);
            }
            let mut iptr = intermediate.as_ptr();
            let mut tmp = tmp;
            for r in 0..#height {
              #inner2
              iptr = iptr.add(1);
              tmp = tmp.add(#width);
            }
          }
        });
      }
    }

    out
  }
}

fn put_8tap_kernel(
  out: &mut TokenStream, frac: &EightTapFrac, feature: &IsaFeature,
  px: PixelType, block: Block,
) -> Ident {
  let body = frac.put_body(px, block);

  feature.to_tokens(out);

  let fn_name_str = format!(
    "put_8tap_{}_{}_{}_{}",
    frac.fn_suffix(),
    block.fn_suffix(),
    feature.fn_suffix(),
    px.type_str()
  );
  let fn_name = Ident::new(&fn_name_str, Span::call_site());
  out.extend(quote! {
    #[inline(never)]
    #[allow(unused_variables)]
    #[allow(unused_mut)]
    #[allow(unused_assignments)]
    pub unsafe fn #fn_name(mut dst: &mut #px, dst_stride: u32,
                           mut src: &#px, src_stride: u32,
                           x_filter: i32x8, y_filter: i32x8,
                           bit_depth: u8, ) {
      let mut dst = dst as *mut #px;
      let mut src = src as *const #px;
      #body
    }
  });

  fn_name
}

fn prep_8tap_kernel(
  out: &mut TokenStream, frac: &EightTapFrac, feature: &IsaFeature,
  px: PixelType, block: Block,
) -> Ident {
  let body = frac.prep_body(px, block);

  feature.to_tokens(out);

  let fn_name_str = format!(
    "prep_8tap_{}_{}_{}_{}",
    frac.fn_suffix(),
    block.fn_suffix(),
    feature.fn_suffix(),
    px.type_str()
  );
  let fn_name = Ident::new(&fn_name_str, Span::call_site());
  out.extend(quote! {
    #[inline(never)]
    #[allow(unused_variables)]
    #[allow(unused_mut)]
    #[allow(unused_assignments)]
    pub unsafe fn #fn_name(tmp: &mut i16,
                           src: &#px, src_stride: u32,
                           x_filter: i32x8, y_filter: i32x8,
                           bit_depth: u8, ) {
      let mut tmp = tmp as *mut i16;
      let mut src = src as *const #px;
      #body
    }
  });

  fn_name
}

fn mc_avg_kernel(
  out: &mut TokenStream, feature: &IsaFeature, px: PixelType, block: Block,
) -> Ident {
  feature.to_tokens(out);

  let fn_name_str = format!(
    "mc_avg_{}_{}_{}",
    block.fn_suffix(),
    feature.fn_suffix(),
    px.type_str()
  );
  let fn_name = Ident::new(&fn_name_str, Span::call_site());

  let mut body = TokenStream::default();
  let width = block.w();
  let height = block.h();
  let inner_step = if width < PrimType::I16.avx2_width() {
    assert_eq!(PrimType::I16.avx2_width() % width, 0);
    width
  } else {
    PrimType::I16.avx2_width()
  };

  if px == PixelType::U8 {
    body.extend(quote! {
      #[cfg(debug_assertions)]
      {
        if bit_depth != 8 {
          panic!("mv_avg kernel expects 8bit depth on u8 pixels");
        }
      }
    });
  }

  body.extend(quote! {
    let zero = Simd::<[i32; #inner_step]>::splat(0);
  });

  let block_size = width * height;

  if block_size / inner_step <= MAX_UNROLL {
    for r in 0..height {
      if r != 0 {
        body.extend(quote! {
          dst = dst.add(dst_stride);
        });
      }
      for c in (0..width).step_by(inner_step) {
        let idx = r * width + c;
        body.extend(quote! {
          let t1 = from_raw_parts(tmp1.add(#idx), #inner_step);
          let t1 = Simd::<[i16; #inner_step]>::
            from_slice_aligned_unchecked(t1);
          let t1 = Simd::<[i32; #inner_step]>::from_cast(t1);
        });
        body.extend(quote! {
          let t2 = from_raw_parts(tmp2.add(#idx), #inner_step);
          let t2 = Simd::<[i16; #inner_step]>::
            from_slice_aligned_unchecked(t2);
          let t2 = Simd::<[i32; #inner_step]>::from_cast(t2);
        });
        body.extend(quote! {
          let t = t1 + t2;
          let t = t.round_shift(intermediate_bits + 1);
        });
        if px == PixelType::U8 {
          // here, we just need to ensure negative numbers
          // don't get wrapped to positive values.
          // XXX assumes u8 <=> 8-bit depth.
          body.extend(quote! {
            let t = t.max(zero);
          });
        } else {
          body.extend(quote! {
            let t = t.clamp(0, max_sample_val);
          });
        }
        let dst = if c != 0 { quote!(dst.add(#c)) } else { quote!(dst) };
        body.extend(quote! {
          let t = Simd::<[#px; #inner_step]>::from_cast(t);
          t.write_to_slice_unaligned_unchecked({
            from_raw_parts_mut(#dst, #inner_step)
          });
        });
      }
    }
  } else {
    // inner_step will always be 8
    let inner_size = width.min(MAX_UNROLL);
    let unrolled_rows = (MAX_UNROLL / inner_size).max(1);

    let mut inner = TokenStream::default();

    for r in 0..unrolled_rows {
      let dst_base = if r != 0 {
        quote! { #r * dst_stride + }
      } else {
        quote! {}
      };
      for c in (0..inner_size).step_by(inner_step) {
        let idx = r * width + c;
        inner.extend(quote! {
          let t1 = from_raw_parts(tmp1.add(#idx), #inner_step);
          let t1 = Simd::<[i16; #inner_step]>::
            from_slice_aligned_unchecked(t1);
          let t1 = Simd::<[i32; #inner_step]>::from_cast(t1);
        });
        inner.extend(quote! {
          let t2 = from_raw_parts(tmp2.add(#idx), #inner_step);
          let t2 = Simd::<[i16; #inner_step]>::
            from_slice_aligned_unchecked(t2);
          let t2 = Simd::<[i32; #inner_step]>::from_cast(t2);
        });
        inner.extend(quote! {
          let t = t1 + t2;
          let t = t.round_shift(intermediate_bits + 1);
        });
        if px == PixelType::U8 {
          // here, we just need to ensure negative numbers
          // don't get wrapped to positive values.
          // XXX assumes u8 <=> 8-bit depth.
          inner.extend(quote! {
            let t = t.max(zero);
          });
        } else {
          inner.extend(quote! {
            let t = t.clamp(0, max_sample_val);
          });
        }
        let dst = if r == 0 && c == 0 {
          quote!(dst)
        } else {
          quote!(dst.add(#dst_base #c))
        };
        inner.extend(quote! {
          let t = Simd::<[#px; #inner_step]>::from_cast(t);
          t.write_to_slice_unaligned_unchecked({
            from_raw_parts_mut(#dst, #inner_step)
          });
        });
      }
    }
    let iter = if unrolled_rows > 1 {
      quote!((0..#height).step_by(#unrolled_rows))
    } else {
      quote!(0..#height)
    };
    if MAX_UNROLL < width {
      assert_eq!(width % MAX_UNROLL, 0);
      let col_reroll = width / MAX_UNROLL;
      assert_ne!(col_reroll, 0);

      let mut t = TokenStream::default();
      for _ in 0..col_reroll {
        t.extend(quote! {
          #inner

          dst = dst.add(#MAX_UNROLL);
          tmp1 = tmp1.add(#MAX_UNROLL);
          tmp2 = tmp2.add(#MAX_UNROLL);
        });
      }

      body.extend(quote! {
        for _ in #iter {
          let dst_start = dst;

          #t

          dst = dst_start.add(dst_stride);
        }
      });
    } else {
      body.extend(quote! {
        for _ in #iter {

          #inner

          dst = dst.add(dst_stride);
          tmp1 = tmp1.add(#width);
          tmp2 = tmp2.add(#width);
        }
      });
    }
  }

  out.extend(quote! {
    #[inline(never)]
    #[allow(unused_variables)]
    #[allow(unused_mut)]
    #[allow(unused_assignments)]
    pub unsafe fn #fn_name(dst: &mut #px, dst_stride: u32,
                           tmp1: &i16, tmp2: &i16,
                           bit_depth: u8, ) {
      let mut dst = dst as *mut #px;
      let mut tmp1 = tmp1 as *const i16;
      let mut tmp2 = tmp2 as *const i16;
      let dst_stride = dst_stride as usize;
      let max_sample_val = ((1 << bit_depth) - 1) as i32;
      let intermediate_bits = 4 - if bit_depth == 12 { 2 } else { 0 };

      #body
    }
  });

  fn_name
}

pub(super) fn put_8tap_kernels(file: &mut dyn Write) {
  write_prelude(file);

  let args = vec![
    quote!(dst: &mut T),
    quote!(dst_stride: u32),
    quote!(src: &T),
    quote!(src_stride: u32),
    quote!(x_filter: i32x8),
    quote!(y_filter: i32x8),
    quote!(bd: u8),
  ];
  let ret = None;
  let mut kernels =
    KernelSet::new("Put8TapF", &args, ret, "PUT_8TAP", vec![quote!(2); 2]);

  for isa in IsaFeature::sets() {
    let mut isa_module = Module::new_root(isa.module_name());
    for px in PixelType::types_iter() {
      let mut px_module = isa_module.new_child(px.module_name());
      for frac in EightTapFrac::fracs().iter() {
        let mut frac_module =
          px_module.new_child_file("put_8tap", frac.module_name());
        StdImports.to_tokens(&mut frac_module);
        for block in Block::blocks_iter() {
          let n = put_8tap_kernel(&mut frac_module, &frac, &isa, px, block);
          let feature_idx = isa.index();
          let b_enum = block.table_idx();
          let row_frac = frac.0 as usize;
          let col_frac = frac.1 as usize;
          let idx = quote! {
            [#feature_idx][#b_enum][#row_frac][#col_frac]
          };
          let path = frac_module.item_path(&n);
          kernels.push_kernel(px, idx, path);
        }
        frac_module.finish_child(&mut px_module);
      }
      px_module.finish_child(&mut isa_module);
    }
    isa_module.finish_root(file);
  }

  println!("generated {} put_8tap kernels", kernels.len());

  let tables = kernels.tables();
  writeln!(file, "{}", tables).expect("write put_8tap kernel tables");
}
pub(super) fn prep_8tap_kernels(file: &mut dyn Write) {
  write_prelude(file);

  let args = vec![
    quote!(tmp: &mut i16),
    quote!(src: &T),
    quote!(stride: u32),
    quote!(x_filter: i32x8),
    quote!(y_filter: i32x8),
    quote!(bd: u8),
  ];
  let ret = None;
  let mut kernels =
    KernelSet::new("Prep8TapF", &args, ret, "PREP_8TAP", vec![quote!(2); 2]);

  for isa in IsaFeature::sets() {
    let mut isa_module = Module::new_root(isa.module_name());
    for px in PixelType::types_iter() {
      let mut px_module = isa_module.new_child(px.module_name());
      for frac in EightTapFrac::fracs().iter() {
        let mut frac_module =
          px_module.new_child_file("prep_8tap", frac.module_name());
        StdImports.to_tokens(&mut frac_module);
        for block in Block::blocks_iter() {
          let n = prep_8tap_kernel(&mut frac_module, &frac, &isa, px, block);
          let feature_idx = isa.index();
          let b_enum = block.table_idx();
          let row_frac = frac.0 as usize;
          let col_frac = frac.1 as usize;
          let idx = quote! {
            [#feature_idx][#b_enum][#row_frac][#col_frac]
          };
          let path = frac_module.item_path(&n);
          kernels.push_kernel(px, idx, path);
        }
        frac_module.finish_child(&mut px_module);
      }
      px_module.finish_child(&mut isa_module);
    }
    isa_module.finish_root(file);
  }

  println!("generated {} prep_8tap kernels", kernels.len());

  let tables = kernels.tables();
  writeln!(file, "{}", tables).expect("write prep_8tap kernel tables");
}
pub(super) fn mc_avg_kernels(file: &mut dyn Write) {
  write_prelude(file);

  let args = vec![
    quote!(dst: &mut T),
    quote!(dst_stride: u32),
    quote!(tmp1: &i16),
    quote!(tmp2: &i16),
    quote!(bit_depth: u8),
  ];
  let ret = None;
  let mut kernels = KernelSet::new("McAvgF", &args, ret, "MC_AVG", vec![]);

  for isa in IsaFeature::sets() {
    let mut isa_module = Module::new_root(isa.module_name());
    for px in PixelType::types_iter() {
      let mut px_module = isa_module.new_child(px.module_name());
      StdImports.to_tokens(&mut px_module);
      for block in Block::blocks_iter() {
        let n = mc_avg_kernel(&mut px_module, &isa, px, block);
        let feature_idx = isa.index();
        let b_enum = block.table_idx();
        let idx = quote! {
          [#feature_idx][#b_enum]
        };
        let path = px_module.item_path(&n);
        kernels.push_kernel(px, idx, path);
      }
      px_module.finish_child(&mut isa_module);
    }
    isa_module.finish_root(file);
  }

  println!("generated {} mc_avg kernels", kernels.len());

  let tables = kernels.tables();
  writeln!(file, "{}", tables).expect("write mc_avg kernel tables");
}
