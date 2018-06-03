# BGHMatcher

Experiments with the Generalized Hough transform.

Instead of using X and Y gradients or Sobel operators as in the classic Generalized Hough algorithm,
this project uses "binary gradients" (for lack of a better name).

Creating a "binary gradient" image is a fast operation with integer math and simple comparisons.
Each pixel is compared with its 8 neighbors.  If the pixel is greater than a neighbor, a bit is set.
This produces an 8-bit code for each pixel.  In practice, a code with 4 or 5 adjacent "1" bits
is a good substitute for the gradient information used in the classic algorithm.

The 8-bit codes make it possible to do a quick lookup to find an array of voting points
associated with each "binary gradient" value.
