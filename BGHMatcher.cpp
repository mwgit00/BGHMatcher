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

#include <map>
#include <list>
#include "BGHMatcher.h"


void BGHMatcher::create_adjacent_bits_set(const uint8_t mask, BGHMatcher::T_256_flags& rflags)
{
    for (int i = 0; i < 8; i++)
    {
        if (mask & (1 << i))
        {
            uint8_t num_bits_in_mask = i + 1;
            uint8_t bit_mask = (1 << num_bits_in_mask) - 1;
            for (int j = 0; j < 8; j++)
            {
                uint8_t rot_bit_mask_L = (bit_mask << j);
                uint8_t rot_bit_mask_R = (bit_mask >> (8 - j));
                uint8_t rot_bit_mask = rot_bit_mask_L | rot_bit_mask_R;
                rflags.set(rot_bit_mask);
            }
        }
    }
}


void BGHMatcher::create_ghough_data(
    const cv::Mat& rsrc,
    const int kblur,
    const BGHMatcher::T_256_flags& rflags,
    BGHMatcher::T_ghough_data& rdata)
{
    cv::Mat img_temp;
    cv::Mat img_bgrad;

    // apply pre-blur to template source image
    GaussianBlur(rsrc, img_temp, { kblur, kblur }, 0);

    // create binary gradient image
    cmp8NeighborsGT<uint8_t>(img_temp, img_bgrad);

    // calculate centering offset
    int row_offset = img_bgrad.rows / 2;
    int col_offset = img_bgrad.cols / 2;

    // use STL structures to build a lookup table dynamically
    std::map<uint8_t, std::list<cv::Point>> lookup_table;
    for (int i = 0; i < img_bgrad.rows; i++)
    {
        const uint8_t * pix = img_bgrad.ptr<uint8_t>(i);
        for (int j = 0; j < img_bgrad.cols; j++)
        {
            const uint8_t uu = pix[j];
            if (rflags.get(uu))
            {
                lookup_table[uu].push_back(cv::Point(col_offset - j, row_offset - i));
            }
        }
    }

    // then put lookup table into a fixed non-STL structure
    // that's much more efficient for debugging
    rdata.sz = rsrc.size();
    rdata.kblur = kblur;
    for (const auto& r : lookup_table)
    {
        uint8_t key = r.first;
        size_t n = r.second.size();
        rdata.elem[key].ct = n;
        rdata.elem[key].pts = new cv::Point[n];
        size_t k = 0;
        for (const auto& rr : r.second)
        {
            rdata.elem[key].pts[k++] = rr;
        }
        rdata.total += n;
    }
}


void BGHMatcher::apply_ghough_transform(
    const BGHMatcher::T_ghough_data& rdata,
    const cv::Mat& rimg,
    cv::Mat& rout)
{
    rout = cv::Mat::zeros(rimg.size(), CV_32F);
    for (int i = rdata.sz.height / 2; i < rimg.rows - rdata.sz.height / 2; i++)
    {
        const uint8_t * pix = rimg.ptr<uint8_t>(i);
        for (int j = rdata.sz.width / 2; j < rimg.cols - rdata.sz.width / 2; j++)
        {
            uint8_t uu = pix[j];
            const size_t ct = rdata.elem[uu].ct;
            for (size_t k = 0; k < ct; k++)
            {
                const cv::Point& rp = rdata.elem[uu].pts[k];
                int mx = (j + rp.x);
                int my = (i + rp.y);
                float * pix = rout.ptr<float>(my) + mx;
                *pix += 1.0;
            }
        }
    }
}
