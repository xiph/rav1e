# Tile on a Frame boundary

## Accessing Tiled-pixels outside of a frame

Tiled input and output planes (type PlaneRegion) are configured in order to
allow memory access of its pixels that are located outside of frame boundary.
This configuration is only effective when the tile is adjacent to right or bottom frame boundary and
a partition block straddles on any frame boundary.

The type names of tiled input and output in the codebase is defined in:

```
pub struct TileStateMut<'a, T: Pixel> {
  ...
  pub input: &'a Frame<T>,     // the whole frame
  pub rec: TileMut<'a, T>,
  ...
}
```

## How bounding box is enforced to a PlaneRegion and now it allows to access outside of frame pixels

A memory access to tiled input and output pixels in a _tile_ 
is bounded by 'rect' field of each plane (of type `PlaneRegion`) of the `input_tile` or `rec` of `TileStateMut`.

The bounding box dimension of a PlaneRegion.rect, i.e. PlaneRegion.rect.width and .height is ROUNDED UP to SuperBlock size,
which now allows accessing the pixels outside the right or bottom frame boundary and belongs to a partition that straddle on any frame boundary.

Previously, the bounding box has prohibited accessing those pixels. 

## When does rav1e access outside frame tiled-input and output pixels?

A CfL requires to read outside the coded frame (luma only), when it computes average
and obtain ac components for luma block, i.e. `luma_ac()`.
Hence, the prediction of a luma plane should be performed for all the pixels in a block
(more precisely, all required tx-blocks in a partition), whether the predicted pixel is inside or outside the frame.
