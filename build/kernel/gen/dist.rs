use std::env;
use std::fs::File;
use std::io::{BufWriter, Cursor, Write};
use std::path::Path;

use super::*;

fn sad_kernel(
  dst: &mut Module, b: Block, px: PixelType, feature: IsaFeature,
) -> (TokenStream, TokenStream) {
  let h = b.h();

  feature.to_tokens(&mut *dst);

  let fn_name_str = format!(
    "sad_{}_{}_{}",
    b.fn_suffix(),
    px.type_str(),
    feature.fn_suffix(),
  );
  let fn_name = Ident::new(&fn_name_str, Span::call_site());

  let mut body = TokenStream::default();

  let sum = Var::new_mut("sum", 0u32);
  sum.to_tokens(&mut body);

  let left_ptr = Ident::new("left", Span::call_site());
  let left = Plane::new(&left_ptr);
  let right_ptr = Ident::new("right", Span::call_site());
  let right = Plane::new(&right_ptr);

  let calc_prim = PrimType::I32; // TODO use i16 if possible
  let src_lanes = 16usize.min(px.avx2_width());
  let lanes = b.w().min(src_lanes);

  let load_simd = SimdType::new(px.into(), lanes);
  let calc_simd = SimdType::new(calc_prim, lanes);
  let hsum = PrimType::U32;

  if b.w() <= src_lanes || b.area() / src_lanes <= MAX_UNROLL {
    // completely unroll the small blocks (ie those <= 8).
    for r in 0..b.h() {
      for c in (0..b.w()).step_by(lanes) {
        let left_ptr = Var::let_(&mut body, "lptr", left.add_rc(r, c));
        let right_ptr = Var::let_(&mut body, "rptr", right.add_rc(r, c));

        let l = load_simd.uload(&quote!(#left_ptr)).let_(&mut body, "l");
        let r = load_simd.uload(&quote!(#right_ptr)).let_(&mut body, "r");
        let l = l.cast(calc_simd).let_(&mut body, "l");
        let r = r.cast(calc_simd).let_(&mut body, "r");
        let abs = (&l - &r).abs().let_(&mut body, "abs");
        sum.add_assign(&mut body, quote!(#abs.wrapping_sum() as #hsum));
      }
    }
  // the remaining else branches only partially unroll.
  } else if src_lanes <= b.w() {
    let mut check = UnrollCheck::new(&fn_name);
    // We have to possibly unroll multiple row iterations.
    let unrolled_rows = MAX_UNROLL / (b.w() / lanes);

    let iter = if unrolled_rows > 1 {
      quote!((0..#h).step_by(#unrolled_rows))
    } else {
      quote!(0..#h)
    };

    let mut inner = TokenStream::default();

    for r in 0..unrolled_rows {
      for c in (0..b.w()).step_by(lanes) {
        let left_ptr = Var::let_(&mut inner, "lptr", left.add_rc(r, c));
        let right_ptr = Var::let_(&mut inner, "rptr", right.add_rc(r, c));

        let l = load_simd.uload(&quote!(#left_ptr)).let_(&mut inner, "l");
        let r = load_simd.uload(&quote!(#right_ptr)).let_(&mut inner, "r");
        let l = l.cast(calc_simd).let_(&mut inner, "l");
        let r = r.cast(calc_simd).let_(&mut inner, "r");
        let abs = (&l - &r).abs().let_(&mut inner, "abs");
        sum.add_assign(&mut inner, quote!(#abs.wrapping_sum() as #hsum));

        check.check();
      }
    }
    check.finish();

    left.next_nth(unrolled_rows, &mut inner);
    right.next_nth(unrolled_rows, &mut inner);

    body.extend(quote! {
      for _ in #iter {
        #inner
      }
    });
  } else {
    unimplemented!("b: {:?}, px: PixelType::{:?}", b, px);
  }

  dst.extend(quote! {
    #[inline(never)]
    #[allow(unused_variables)]
    #[allow(unused_mut)]
    #[allow(unused_assignments)]
    pub unsafe fn #fn_name(left: &#px, left_stride: u32,
                           right: &#px, right_stride: u32)
      -> u32
    {
      let mut left = left as *const #px;
      let mut right = right as *const #px;
      let left_stride = left_stride as usize;
      let right_stride = right_stride as usize;

      #body

      sum
    }
  });

  let feature_idx = feature.index();
  let b_enum = b.table_idx();
  let idx = quote! {
    [#feature_idx][#b_enum]
  };
  let path = dst.item_path(&fn_name);

  (idx, path)
}

pub(super) fn sad_kernels(file: &mut dyn Write) {
  write_prelude(file);

  let args = vec![
    quote!(left: &T),
    quote!(left_stride: u32),
    quote!(right: &T),
    quote!(right_stride: u32),
  ];
  let ret = quote!(-> u32);
  let mut kernels = KernelSet::new("SadF", &args, Some(ret), "SAD", vec![]);

  for isa in IsaFeature::sets() {
    let mut isa_module = Module::new_root(isa.module_name());
    for px in PixelType::types_iter() {
      let mut px_module = isa_module.new_child_file("sad", px.module_name());
      StdImports.to_tokens(&mut px_module);
      for block in Block::blocks_iter() {
        let (idx, path) = sad_kernel(&mut px_module, block, px, isa);
        kernels.push_kernel(px, idx, path);
      }
      px_module.finish_child(&mut isa_module);
    }
    isa_module.finish_root(file);
  }

  println!("generated {} sad kernels", kernels.len());

  let tables = kernels.tables();
  writeln!(file, "{}", tables).expect("write sad kernel tables");
}

fn butterfly_let<T, U>(
  dst: &mut TokenStream, l: &SimdValue, r: &SimdValue, ln: T, rn: U,
) -> (SimdValue, SimdValue)
where
  T: Display,
  U: Display,
{
  let (l, r) = l.butterfly(r);
  let l = l.let_(dst, ln);
  let r = r.let_(dst, rn);
  (l, r)
}
fn satd_4nx4n_kernel<T>(
  dst: &mut TokenStream, sum: &Var<T>, px: PixelType, porg: &Plane,
  pref: &Plane,
) where
  T: ToTokens,
{
  const LANES: usize = 4usize;

  const FIRST_HALF: [u32; 8] = [0, 1, 2, 3, 8, 9, 10, 11];
  const SECOND_HALF: [u32; 8] = [4, 5, 6, 7, 12, 13, 14, 15];

  const TRANSPOSE_IDX1: [u32; 8] = [0, 8, 4, 12, 2, 10, 6, 14];
  const TRANSPOSE_IDX2: [u32; 8] = [1, 9, 5, 13, 3, 11, 7, 15];

  let load_ty = SimdType::new(px.into(), LANES);
  // TODO < 16 bit depth?
  let _calc_prim = match px {
    PixelType::U8 => PrimType::I16,
    PixelType::U16 => PrimType::I32,
  };
  let calc_prim = PrimType::I16;
  let calc_ty = SimdType::new(calc_prim, 8);

  let a =
    (0..4usize).map(|i| load_ty.uload(&porg.add_rc(i, 0usize))).collect();
  let a = Vector::new(load_ty, a);
  let a = VarArr::let_mut(dst, "a", &a);
  let b =
    (0..4usize).map(|i| load_ty.uload(&pref.add_rc(i, 0usize))).collect();
  let b = Vector::new(load_ty, b);
  let b = VarArr::let_mut(dst, "b", &b);

  let a02 = a.get(0).concat(&a.get(2)).let_(dst, "a02");
  let a02 = a02.cast(calc_ty).let_(dst, "a02");
  let a13 = a.get(1).concat(&a.get(3)).let_(dst, "a13");
  let a13 = a13.cast(calc_ty).let_(dst, "a13");

  let b02 = b.get(0).concat(&b.get(2)).let_(dst, "b02");
  let b02 = b02.cast(calc_ty).let_(dst, "b02");
  let b13 = b.get(1).concat(&b.get(3)).let_(dst, "b13");
  let b13 = b13.cast(calc_ty).let_(dst, "b13");

  let ab02 = (a02 - b02).let_(dst, "ab02");
  let ab13 = (a13 - b13).let_(dst, "ab13");

  let (a0a2, a1a3) = butterfly_let(dst, &ab02, &ab13, "a0a2", "a1a3");
  let a0a1 = SimdValue::shuffle2(&a0a2, &a1a3, &FIRST_HALF).let_(dst, "a0a1");
  let a2a3 = SimdValue::shuffle2(&a0a2, &a1a3, &SECOND_HALF).let_(dst, "a2a3");
  let (b0b2, b1b3) = butterfly_let(dst, &a0a1, &a2a3, "b0b2", "b1b3");

  let t0 = SimdValue::shuffle2(&b0b2, &b1b3, &TRANSPOSE_IDX1).let_(dst, "t0");
  let t1 = SimdValue::shuffle2(&b0b2, &b1b3, &TRANSPOSE_IDX2).let_(dst, "t1");

  let (a0a2, a1a3) = butterfly_let(dst, &t0, &t1, "a0a2", "a1a3");
  let a0a1 =
    SimdValue::shuffle2(&a0a2, &a1a3, { &[0u32, 1, 2, 3, 8, 9, 10, 11] })
      .let_(dst, "a0a1");
  let a2a3 =
    SimdValue::shuffle2(&a0a2, &a1a3, { &[4u32, 5, 6, 7, 12, 13, 14, 15] })
      .let_(dst, "a2a3");

  // Use the fact that
  //   (abs(a+b)+abs(a-b))/2 = max(abs(a),abs(b))
  // to merge the final butterfly with the abs and the first stage of
  // accumulation.
  let c0 = a0a1.abs().max(&a2a3.abs()) * calc_ty.splat(2);
  let c0 = c0.cast(SimdType::new(PrimType::U32, 8)).let_(dst, "c0");
  sum.add_assign(dst, quote!(#c0.wrapping_sum()));
}
fn satd_8nx8n_kernel<T>(
  dst: &mut TokenStream, sum: &Var<T>, px: PixelType, porg: &Plane,
  pref: &Plane,
) where
  T: ToTokens,
{
  const LANES: usize = 8usize;

  let load_ty = SimdType::new(px.into(), LANES);
  // TODO < 16 bit depth?
  let _calc_prim = match px {
    PixelType::U8 => PrimType::I16,
    PixelType::U16 => PrimType::I32,
  };
  let calc_prim = PrimType::I16;
  let calc_ty = SimdType::new(calc_prim, 8);

  let a = (0..8usize)
    .map(|i| load_ty.uload(&porg.add_rc(i, 0usize)).cast(calc_ty))
    .collect();
  let a = Vector::new(calc_ty, a);
  let a = VarArr::let_(dst, "a", &a);
  let b = (0..8usize)
    .map(|i| load_ty.uload(&pref.add_rc(i, 0usize)).cast(calc_ty))
    .collect();
  let b = Vector::new(calc_ty, b);
  let b = VarArr::let_(dst, "b", &b);

  let ab = (0..8).map(|i| a.get(i) - b.get(i)).collect();
  let ab = Vector::new(calc_ty, ab);
  let ab = VarArr::let_(dst, "ab", &ab);

  let (a0, a1) = butterfly_let(dst, &ab.get(0), &ab.get(1), "a0", "a1");
  let (a2, a3) = butterfly_let(dst, &ab.get(2), &ab.get(3), "a2", "a3");
  let (a4, a5) = butterfly_let(dst, &ab.get(4), &ab.get(5), "a4", "a5");
  let (a6, a7) = butterfly_let(dst, &ab.get(6), &ab.get(7), "a6", "a7");

  let (b0, b2) = butterfly_let(dst, &a0, &a2, "b0", "b1");
  let (b1, b3) = butterfly_let(dst, &a1, &a3, "b2", "b3");
  let (b4, b6) = butterfly_let(dst, &a4, &a6, "b4", "b5");
  let (b5, b7) = butterfly_let(dst, &a5, &a7, "b6", "b7");

  let (c0, c4) = b0.butterfly(&b4);
  let (c1, c5) = b1.butterfly(&b5);
  let (c2, c6) = b2.butterfly(&b6);
  let (c3, c7) = b3.butterfly(&b7);

  let c0 = c0.let_mut(dst, "c0");
  let c1 = c1.let_mut(dst, "c1");
  let c2 = c2.let_mut(dst, "c2");
  let c3 = c3.let_mut(dst, "c3");
  let c4 = c4.let_mut(dst, "c4");
  let c5 = c5.let_mut(dst, "c5");
  let c6 = c6.let_mut(dst, "c6");
  let c7 = c7.let_mut(dst, "c7");

  let c = vec![&c0, &c1, &c2, &c3, &c4, &c5, &c6, &c7];

  // Transpose
  let mut transpose = TokenStream::default();
  for i in 0..8 {
    for j in 0..i {
      let l = &c[j];
      let r = &c[i];
      transpose.extend(quote! {
        let l = #l.extract_unchecked(#i);
        let r = #r.extract_unchecked(#j);
        #l = #l.replace_unchecked(#i, r);
        #r = #r.replace_unchecked(#j, l);
      });
    }
  }
  // wrap in a block so it can be collapsed by editors.
  dst.extend(quote!({ #transpose }));

  let (a0, a1) = butterfly_let(dst, &c0, &c1, "a0", "a1");
  let (a2, a3) = butterfly_let(dst, &c2, &c3, "a2", "a3");
  let (a4, a5) = butterfly_let(dst, &c4, &c5, "a4", "a5");
  let (a6, a7) = butterfly_let(dst, &c6, &c7, "a6", "a7");

  let (b0, b2) = butterfly_let(dst, &a0, &a2, "b0", "b2");
  let (b1, b3) = butterfly_let(dst, &a1, &a3, "b1", "b3");
  let (b4, b6) = butterfly_let(dst, &a4, &a6, "b4", "b6");
  let (b5, b7) = butterfly_let(dst, &a5, &a7, "b5", "b7");

  // Use the fact that
  //   (abs(a+b)+abs(a-b))/2 = max(abs(a),abs(b))
  // to merge the final butterfly with the abs and the first stage of
  // accumulation.
  //
  // What on Earth does this mean:
  // Avoid pabsw by using max(a, b) + max(a + b + 0x7FFF, 0x7FFF) instead.
  // Actually calculates (abs(a+b)+abs(a-b))/2-0x7FFF.
  // The final sum must be offset to compensate for subtracting 0x7FFF.

  let two = calc_ty.splat(2);
  let c0 = (b0.abs().max(&b4.abs()) * &two).let_(dst, "c0");
  let c1 = (b1.abs().max(&b5.abs()) * &two).let_(dst, "c1");
  let c2 = (b2.abs().max(&b6.abs()) * &two).let_(dst, "c2");
  let c3 = (b3.abs().max(&b7.abs()) * &two).let_(dst, "c3");

  let sum_ty = SimdType::new(PrimType::U32, LANES);
  let d0 = c0.cast(sum_ty).let_(dst, "d0");
  let d1 = c1.cast(sum_ty).let_(dst, "d1");
  let d2 = c2.cast(sum_ty).let_(dst, "d2");
  let d3 = c3.cast(sum_ty).let_(dst, "d3");

  sum.add_assign(dst, quote!(#d0.wrapping_sum()));
  sum.add_assign(dst, quote!(#d1.wrapping_sum()));
  sum.add_assign(dst, quote!(#d2.wrapping_sum()));
  sum.add_assign(dst, quote!(#d3.wrapping_sum()));
}

/// These kernels are not just a few instructions; prevent us from
/// unrolling much at all.
const SATD_MAX_UNROLL: usize = 4;

fn satd_kernel(
  dst: &mut Module, b: Block, px: PixelType, feature: IsaFeature,
) -> (TokenStream, TokenStream) {
  let w = b.w();
  let h = b.h();

  feature.to_tokens(&mut *dst);

  let fn_name_str = format!(
    "satd_{}_{}_{}",
    b.fn_suffix(),
    px.type_str(),
    feature.fn_suffix(),
  );
  let fn_name = Ident::new(&fn_name_str, Span::call_site());

  let mut body = TokenStream::default();

  let sum = Var::new_mut("sum", 0u32);
  sum.to_tokens(&mut body);

  let porg_ptr = Ident::new("porg", Span::call_site());
  let porg = Plane::new(&porg_ptr);
  let pref_ptr = Ident::new("pref", Span::call_site());
  let pref = Plane::new(&pref_ptr);

  let size = w.min(h).min(8);
  let step = size;

  let max_unroll = if size == 4 {
    SATD_MAX_UNROLL
  } else {
    // unroll these less
    2
  };

  if (b.area() / step) / step <= max_unroll {
    // completely unroll the small blocks
    for r in (0..b.h()).step_by(step) {
      for c in (0..b.w()).step_by(step) {
        let porg_ptr = Var::let_(&mut body, "lporg", porg.add_rc(r, c));
        let pref_ptr = Var::let_(&mut body, "lpref", pref.add_rc(r, c));
        let porg = Plane::new_stride(&porg_ptr, "porg_stride");
        let pref = Plane::new_stride(&pref_ptr, "pref_stride");

        if step == 4 {
          satd_4nx4n_kernel(&mut body, &sum, px, &porg, &pref)
        } else {
          assert_eq!(step, 8);
          satd_8nx8n_kernel(&mut body, &sum, px, &porg, &pref);
        }
      }
    }
  // the remaining else branches only partially unroll.
  } else if b.w() / step > max_unroll {
    // multiple iterations per column
    let mut check = UnrollCheck::new_max(&fn_name, max_unroll);

    let unrolled_cols = max_unroll;

    let iter = if unrolled_cols > 1 {
      quote!((0..#w).step_by(#step * #unrolled_cols))
    } else {
      quote!((0..#w).step_by(#step))
    };

    let mut inner = TokenStream::default();

    let cporg_ptr = Ident::new("cporg", Span::call_site());
    let cpref_ptr = Ident::new("cpref", Span::call_site());
    let cporg = Plane::new_stride(&cporg_ptr, "porg_stride");
    let cpref = Plane::new_stride(&cpref_ptr, "pref_stride");
    for c in 0..unrolled_cols {
      let porg_ptr = Var::let_(&mut inner, "lporg", cporg.add(step * c));
      let pref_ptr = Var::let_(&mut inner, "lpref", cpref.add(step * c));
      let porg = Plane::new_stride(&porg_ptr, "porg_stride");
      let pref = Plane::new_stride(&pref_ptr, "pref_stride");

      if step == 4 {
        satd_4nx4n_kernel(&mut inner, &sum, px, &porg, &pref)
      } else {
        assert_eq!(step, 8);
        satd_8nx8n_kernel(&mut inner, &sum, px, &porg, &pref);
      }

      check.check();
    }
    check.finish();

    let mut next_row = TokenStream::default();
    porg.next_nth(step, &mut next_row);
    pref.next_nth(step, &mut next_row);
    next_row.extend(quote! {
      #cporg_ptr = #porg;
      #cpref_ptr = #pref;
    });

    body.extend(quote! {
      let mut r_i = (0..#h).step_by(#step);
      r_i.next();
      let mut c_i = #iter;
      c_i.next();
      let mut #cporg_ptr = #porg;
      let mut #cpref_ptr = #pref;
      loop {
        #inner

        if let Some(_) = c_i.next() {
          #cporg_ptr = #cporg_ptr.add(#step * #unrolled_cols);
          #cpref_ptr = #cpref_ptr.add(#step * #unrolled_cols);
          continue;
        } else if let Some(_) = r_i.next() {
          #next_row
          c_i = #iter;
          c_i.next();
          continue;
        } else {
          break;
        }
      }
    });
  } else {
    // multiple iterations per row
    let mut check = UnrollCheck::new_max(&fn_name, max_unroll);
    // We have to possibly unroll multiple row iterations.
    let unrolled_rows = max_unroll / (b.w() / step);
    let unrolled_rows = unrolled_rows.max(1);

    let iter = if unrolled_rows > 1 {
      quote!((0..#h).step_by(#step * #unrolled_rows))
    } else {
      quote!((0..#h).step_by(#step))
    };

    let mut inner = TokenStream::default();

    {
      for r in 0..unrolled_rows {
        for c in (0..b.w()).step_by(step) {
          let porg_ptr =
            Var::let_(&mut inner, "lporg", porg.add_rc(r * step, c));
          let pref_ptr =
            Var::let_(&mut inner, "lpref", pref.add_rc(r * step, c));
          let porg = Plane::new_stride(&porg_ptr, "porg_stride");
          let pref = Plane::new_stride(&pref_ptr, "pref_stride");

          if step == 4 {
            satd_4nx4n_kernel(&mut inner, &sum, px, &porg, &pref)
          } else {
            assert_eq!(step, 8);
            satd_8nx8n_kernel(&mut inner, &sum, px, &porg, &pref);
          }

          check.check();
        }
      }
      check.finish();
    }

    let mut next_row = TokenStream::default();
    porg.next_nth(step * unrolled_rows, &mut next_row);
    pref.next_nth(step * unrolled_rows, &mut next_row);

    body.extend(quote! {
      let mut r_i = #iter;
      r_i.next();
      loop {
        #inner

        if let None = r_i.next() {
          break;
        }

        #next_row
      }
    });
  }

  let size = size as i32;
  dst.extend(quote! {
    #[inline(never)]
    #[allow(unused_variables)]
    #[allow(unused_mut)]
    #[allow(unused_assignments)]
    #[allow(unused_parens)]
    pub unsafe fn #fn_name(porg: &#px, porg_stride: u32,
                           pref: &#px, pref_stride: u32)
      -> u32
    {
      let mut porg = porg as *const #px;
      let mut pref = pref as *const #px;
      let porg_stride = porg_stride as usize;
      let pref_stride = pref_stride as usize;

      #body

      // Normalize the results
      let ln = crate::util::msb(#size) as u64;
      ((sum + (1 << ln >> 1)) >> ln) as u32
    }
  });

  let feature_idx = feature.index();
  let b_enum = b.table_idx();
  let idx = quote! {
    [#feature_idx][#b_enum]
  };
  let path = dst.item_path(&fn_name);

  (idx, path)
}

pub(super) fn satd_kernels(file: &mut dyn Write) {
  write_prelude(file);

  let args = vec![
    quote!(porg: &T),
    quote!(porg_stride: u32),
    quote!(pref: &T),
    quote!(pref_stride: u32),
  ];
  let ret = quote!(-> u32);
  let mut kernels = KernelSet::new("SatdF", &args, Some(ret), "SATD", vec![]);

  for isa in IsaFeature::sets() {
    let mut isa_module = Module::new_root(isa.module_name());
    for px in PixelType::types_iter() {
      let mut px_module = isa_module.new_child_file("satd", px.module_name());
      StdImports.to_tokens(&mut px_module);
      for block in Block::blocks_iter() {
        let (idx, path) = satd_kernel(&mut px_module, block, px, isa);
        kernels.push_kernel(px, idx, path);
      }
      px_module.finish_child(&mut isa_module);
    }
    isa_module.finish_root(file);
  }

  println!("generated {} satd kernels", kernels.len());

  let tables = kernels.tables();
  writeln!(file, "{}", tables).expect("write satd kernel tables");
}
