# BGHMatcher

Experiments with the Generalized Hough Transform.

This repository has an implementation of the the classic Generalized Hough (GH) algorithm.
Sobel filters calculate the X and Y derivatives.  A Cartesian-to-Polar operation converts
the X and Y derivatives to magnitude and angle.  Gradients with magnitudes below an
arbitrary threshold are discarded.  The angles are converted to 8-bit integer codes
for quick lookup in a voting table.

Some implementations of the GH algorithm use a Canny Edge Detector as a pre-processing step and
sub-divide an image into voting bins.  It can be difficult to tune the Canny detector and bin size.
In this implementation, the gradient calculations are blurry and the edges are "fuzzy".  This
creates a lot of redundant votes.  In practice, this creates well-defined maxima.  The maxima
are located at single pixels rather than at voting bins, which may be associated with multiple
pixels.  Blurry gradients might also provide more tolerance to variations in scale and
rotation when finding matches in the target image.

# Installation

The project compiles in the Community edition of Visual Studio 2015 (VS 2015).
It uses the Windows pre-built OpenCV 4.0.1 libraries extracted to **c:\opencv-4.0.1**.
I just copied the appropriate OpenCV DLLs to wherever I had my executables.  It creates a
command-line Windows executable.  I have tested it on a Windows 7 64-bit machine with Service Pack 1.

## Camera

I tested with a Logitech c270.  It was the cheapest one I could find that I could purchase locally.
It was plug-and-play.

## Videos

Here is a demo video:

https://www.youtube.com/watch?v=heNQ9mr__L8


