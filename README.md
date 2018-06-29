# BGHMatcher

Experiments with the Generalized Hough Transform.

This repository has an implementation of the the classic Generalized Hough algorithm.
Sobel filters calculate the X and Y derivatives.  A Cartesian-to-Polar operation converts
the X and Y derivatives to magnitude and angle.  Gradients with magnitudes below an
arbitrary threshold are discarded.  The angles are converted to 8-bit integer codes
for quick lookup in a voting table.
