Temporal Unit
---------------
The same as an AV1 temporal unit. In rav1e, a Temporal Unit always starts with a temporal delimiter, followed by zero or one sequence headers, one or more frame headers and zero or more tile groups.

Tile Group
-----------
Same as an AV1 tile group. Zero or more of these, plus a Frame Header, make up a Frame.

Frame Header
--------------
Same as an AV1 frame header. These are followed by a tile group, except when the show-existing-frame feature is used, in which case there are zero tile groups.

Group
------
Otherwise known as a subgop, this is a group of reordered frames. The first frames in the group will be non-shown frames, followed by a sequence of shown frames.  A group may either start with a key frame or inter frame.

Frame
------
In the input, this is one picture of YUV data.
In the bitstream, this is a Frame Header followed by zero or more Tile Groups.
With reordering, there will be more frames in the output than the input, but some will not be shown.
  
Segment
--------
A sequence of groups. The first group will start with a keyframe, and the rest of the groups will start with inter frames.
