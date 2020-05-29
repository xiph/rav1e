# Frame Type Selection

<details>
<summary><b>Table of Content</b></summary>

- [Current Features/Process](#current-featuresprocess)
- [Detection Algorithm](#detection-algorithm)
- [Desired Improvements](#desired-improvements)
</details>

## Current Features/Process
* The first frame of the video is always a key frame.
* rav1e looks ahead up to the maximum number of frames in a sub-GOP
to detect "flashes" of content, which are then marked ineligible for
a scenecut.
* If the user has forced the current frame to be a key frame, it is marked as a key frame.
This overrides all other criteria for frame type selection. (TODO: How does a user do this?)
* If there have been fewer than the number of frames specified by `--min-keyint`
since the last key frame, the current frame is marked as an inter frame.
* If there have been equal to the number of frames specified by `--keyint` (i.e. the max keyint)
since the last key frame, the current frame is marked as a key frame.
* If no other criteria have been met, the current frame is compared with
the previous frame to see if it is a scenecut.
If it is a scenecut, it is marked as a key frame, otherwise it is marked as an inter frame.

## Detection Algorithm
* On speeds 0-9, the algorithm compares frame intra cost vs. inter cost. This is better for compression, but slower.
* On speed 10, the algorithm compares the amount of difference between frames.

## Desired Improvements
* If the max keyint length is in the middle of a flash of content, the key frame should be placed at either the start or end of the flash, instead of in the middle (exactly on the max keyint).