pub fn cast<const N: usize, T>(x: &[T]) -> &[T; N] {
  // SAFETY: we perform a bounds check with [..N],
  // so casting to *const [T; N] is valid because the bounds
  // check guarantees that x has N elements
  unsafe { &*(&x[..N] as *const [T] as *const [T; N]) }
}

pub fn cast_mut<const N: usize, T>(x: &mut [T]) -> &mut [T; N] {
  // SAFETY: we perform a bounds check with [..N],
  // so casting to *mut [T; N] is valid because the bounds
  // check guarantees that x has N elements
  unsafe { &mut *(&mut x[..N] as *mut [T] as *mut [T; N]) }
}
