# BGHMatcher

Experiments with the Generalized Hough Transform.

This repository has an implementation of the the classic Generalized Hough algorithm.
Sobel filters calculate the X and Y derivatives.  A Cartesian-to-Polar operation converts
the X and Y derivatives to magnitude and angle.  Gradients with magnitudes below an
arbitrary threshold are discarded.  The angles are converted to 8-bit integer codes
for quick lookup in a voting table.

# Installation

The project compiles in the Community edition of Visual Studio 2015 (VS 2015).
It uses the Windows pre-built OpenCV 4.0.1 libraries extracted to **c:\opencv-4.0.1**.
I just copied the appropriate OpenCV DLLs to wherever I had my executables.  It creates a
command-line Windows executable.  I have tested it on a Windows 7 64-bit machine with Service Pack 1.

## Camera

I tested with a Logitech c270.  It was the cheapest one I could find that I could purchase locally.  It was plug-and-play.
