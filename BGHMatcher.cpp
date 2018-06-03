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


namespace BGHMatcher
{
    void blur_img(cv::Mat& rsrc, cv::Mat& rdst, const int kblur, const int blur_type)
    {
        switch (blur_type)
        {
        case BGHMatcher::BLUR_GAUSS:
            GaussianBlur(rsrc, rdst, { kblur, kblur }, 0);
            break;
        case BGHMatcher::BLUR_MEDIAN:
            medianBlur(rsrc, rdst, kblur);
            break;
        case BGHMatcher::BLUR_BOX:
        default:
            blur(rsrc, rdst, { kblur, kblur });
            break;
        }
    }


    void create_adjacent_bits_set(
        BGHMatcher::T_256_flags& rflags,
        const uint8_t mask)
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


    void create_ghough_table(
        const cv::Mat& rbgrad,
        const BGHMatcher::T_256_flags& rflags,
        BGHMatcher::T_ghough_table& rtable)
    {
        // calculate centering offset
        int row_offset = rbgrad.rows / 2;
        int col_offset = rbgrad.cols / 2;

        // use STL structures to build a lookup table dynamically
        std::map<uint8_t, std::list<cv::Point>> lookup_table;
        for (int i = 0; i < rbgrad.rows; i++)
        {
            const uint8_t * pix = rbgrad.ptr<uint8_t>(i);
            for (int j = 0; j < rbgrad.cols; j++)
            {
                const uint8_t uu = pix[j];
                if (rflags.get(uu))
                {
                    lookup_table[uu].push_back(cv::Point(col_offset - j, row_offset - i));
                }
            }
        }

        // then put lookup table into a fixed non-STL structure
        // that's much more efficient when running debug code
        rtable.sz = rbgrad.size();
        for (const auto& r : lookup_table)
        {
            uint8_t key = r.first;
            size_t n = r.second.size();
            rtable.elem[key].ct = n;
            rtable.elem[key].pts = new cv::Point[n];
            size_t k = 0;
            for (const auto& rr : r.second)
            {
                rtable.elem[key].pts[k++] = rr;
            }
            rtable.total_votes += n;
        }
    }


    void init_ghough_table_from_img(
        cv::Mat& rimg,
        BGHMatcher::T_ghough_table& rtable,
        const int kblur,
        const int blur_type)
    {
        cv::Mat img_bgrad;
        cv::Mat img_target;

        BGHMatcher::T_256_flags flags;
        BGHMatcher::create_adjacent_bits_set(flags, BGHMatcher::N8_4ADJ);

        // create pre-blurred version of target image
        // same blur should be applied to input image when looking for matches
        blur_img(rimg, img_target, kblur, blur_type);

        // create binary gradient image from blurred target image
        // and create Generalized Hough lookup table from it
        BGHMatcher::cmp8NeighborsGT<uint8_t>(img_target, img_bgrad);
        BGHMatcher::create_ghough_table(img_bgrad, flags, rtable);

        // save metadata for lookup table
        rtable.kblur = kblur;
        rtable.blur_type = blur_type;
    }
}
