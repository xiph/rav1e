# AV1 Encoder Roadmap

Write syntax elements & reconstruct
- Write partitions
- Write ZeroMV / skipmv
- Write fullpel MV
- Write subpel MV
- Write variable sized transforms

RDO
- Breadth-first partition split search
  + Transform size search
    * Transform type search
- Simple hex motion search
  + SAD
  + SATD
  + Model-based R-D
- Luma/Chroma weighting
- Alternate (8x8) distortion functions
- Quant matrices
- Activity masking (spatial_segmentation)
- Quantization/tokenization
  + Trellis optimization
  + Greedy optimization
  + Faster rounding/thresholding strategies
- Loop filters
  + Deblocking
    * Approximate during partition search
  + CDEF
  + Joint parameter optimization
  + Loop Restoration
- Inter modes
  + Compound mode
  + Inter-inter compound segments
  + Inter-intra
  + Warped motion
  + OBMC
  + Wedge modes
  + Global motion
+ Non-causal (iterated/dynamic programming) MV/mode search

Add assembly support
- Port libaom stuffs

RDO speed optimizations
Rate estimation for coefficients (in decreasing order of complexity)
- Estimate rate of coefficients after quantization
- Estimate distortion from residual + quantizer
- Estimate rate of coefficients before quantization
- Estimate rate of coefficients pre inverse transform
- Estimate rate of coefficients from residual

Overall search
- Least accurate search -> list of best possibilities -> slower -> shorter list -> slowest -> pick 1
- Decide size of list item (superblock, block, etc)

Rate control
- Constant quantizer
- Constant bitrate (over a window)
- Real-time/interactive (sub-frame control loop)
- Two-pass
  + Fast first pass
- Look-ahead
- Temporal RDO (MB-Tree)
- Scene change detection
- Frame type decision
  + Pyramid reference structure

=================================================

What do we need to have done to start development:
 - catch up on AV1 bitstream
 - run rav1e on AWCY with normal video sets
 - have a repository URL (gitlab, github?) - Make decision by end of the month, GitHub or GitLab, TD will figure out the exact date.
   - how do we accept anonymous patches
   - look at how GNOME is using gitlab and do that
   - fix github logins
 - decide on auto formatting, style guide, naming conventions, etc.

 ---------------------------------------------------------------------------------------------------------------------------

[MVP](https://github.com/xiph/rav1e/wiki/MVP)





TD-Linux 
Update rav1e to work with unmodified latest libaom
set up gitlab (maybe)
 - set up gitlab ci

xiphmont
lv_map

codeview
partitions

lu_zero
speed

ltrudeau
Add IntraOnly support in Uncompressed Header
Support inter frames

tmatth
partitions

unlord
???

derf
???

barrbrain
cfl

=== RDO Checkpoint Fix ===

Currently, RDO checkpoints make a copy of most context. When the checkpoint is restored, all that context is copied back. This is slow. One idea is for all Writer functions to log what context they touched, and only restore that. However, this still means the copy into the checkpoint has to be everything. That can be made smaller by knowing in advance what context will be touched when creating the checkpoint. However there's another fix:
    
1. Make sure all context is only written by Writer. E.g. partition context must be fixed.
2. Remove checkpoints, and instead add a ContextWriterTransaction (?)
3. cwt = cw.create_transaction();
4. cwt contains a mutable reference to cw so cw can't be used until cwt is dropped
5. all Writer functions are implemented by ContextWriterTransaction. It copies state into itself only when actually mutated.
6. cwt.commit() copies all state back into the cw. Only call this once you want to commit to a mode.
7. cwt can itself create a transaction, e.g. cwt.create_transaction(). This can go arbitrarily deep.