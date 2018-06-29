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

#include "Windows.h"

#include "opencv2/highgui.hpp"

#include <list>
#include <iostream>
#include <sstream>

#include "util.h"


void get_dir_list(
    const std::string& rsdir,
    const std::string& rspattern,
    std::list<std::string>& listOfFiles)
{
    std::string s = rsdir + "\\" + rspattern;

    WIN32_FIND_DATA search_data;
    memset(&search_data, 0, sizeof(WIN32_FIND_DATA));

    HANDLE handle = FindFirstFile(s.data(), &search_data);

    while (handle != INVALID_HANDLE_VALUE)
    {
        std::string sfile(search_data.cFileName);
        listOfFiles.push_back(rsdir + "\\" + sfile);
        if (FindNextFile(handle, &search_data) == FALSE)
        {
            break;
        }
    }

    FindClose(handle);
}


bool make_video(
    const double fps,
    const std::string& rspath,
    const std::string& rsname,
    const int iFOURCC,
    const std::list<std::string>& rListOfPNG)
{
    bool result = false;
    
    // determine size of frames from first image in list
    // they should all be the same size
    const std::string& rs = rListOfPNG.front();
    cv::Mat img = cv::imread(rs);
    cv::Size img_sz = img.size();

    std::string sname = rspath + "\\" + rsname;

    // build movie from separate frames
    cv::VideoWriter vw = cv::VideoWriter(sname, iFOURCC, fps, img_sz);

    if (vw.isOpened())
    {
        for (const auto& r : rListOfPNG)
        {
            cv::Mat img = cv::imread(r);
            vw.write(img);
        }
        
        result = true;
    }

    return result;
}
