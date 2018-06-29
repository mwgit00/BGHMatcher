// MIT License
//
// Copyright(c) 2018 Mark Whitney
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef UTIL_H_
#define UTIL_H_

#include <string>
#include <list>

typedef struct
{
    double mag_thr;
    std::string sname;
} T_file_info;

// Get list of all files in a directory that match a pattern
void get_dir_list(
    const std::string& rsdir,
    const std::string& rspattern,
    std::list<std::string>& listOfFiles);

// Use OpenCV routine to make video from a list of files.
// Here are some extension and FOURCC combos that should work in Windows:
// "movie.wmv", CV_FOURCC('W', 'M', 'V', '2')
// "movie.avi", CV_FOURCC('M', 'J', 'P', 'G')
// "movie.avi", CV_FOURCC('M', 'P', '4', '2')
// "movie.avi", CV_FOURCC('M', 'P', '4', 'V')  -- error messages but VLC can play it
// "movie.mov", CV_FOURCC('M', 'P', '4', 'V')  -- error messages but VLC can play it, iMovie can import it
// "movie.mov", CV_FOURCC('M', 'J', 'P', 'G')  -- error messages but VLC can play it, iMovie can import it
bool make_video(
    const double fps,
    const std::string& rspath,
    const std::string& rsname,
    const int iFOURCC,
    const std::list<std::string>& rListOfPNG);

#endif // UTIL_H_
