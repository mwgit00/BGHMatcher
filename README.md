# BGHMatcher

Experiments with the Generalized Hough Transform.

This repository has an implementation of the the classic Generalized Hough algorithm.
Sobel filters calculate the X and Y derivatives.  A Cartesian-to-Polar operation converts
the X and Y derivatives to magnitude and angle.  Gradients with magnitudes below an
arbitrary threshold are discarded.  The angles are converted to 8-bit integer codes
for quick lookup in a table.

In addition to the classic method, there is some experimental code that uses
"binary gradients" (for lack of a better name).  In these images, each pixel is compared
with its 8 neighbors.  If a pixel is greater than a neighbor, a bit is set.  This
produces an 8-bit integer code for each pixel.  In practice, a code with 4 or 5
adjacent "1" bits is a good substitute for the gradient information used in the 
classic algorithm.  The difference between the maximum and minimum in each 3x3
neighborhood is used as a metric for the magnitude of the gradient.

The "binary gradient" approach compares favorably with the classic method but
its simpler operations can provide a significant speed increase.
