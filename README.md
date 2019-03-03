# horn_schunck_cuda
Quick port from a MATLAB Horn Schunck implementation that was done for a university lecture, implemented as an Adobe CC plugin.
Uses a pyramidal coarse to fine approach.
Tried to get slomo with frame interpolation by optical flow warping working.
Works well on some sequences but not production ready in any way.
Would probably benefit froma robust forward warping implementation opposed to the backwards warping that's used now.

Testvideo can be found here: https://youtu.be/czCJPoWkNvI
