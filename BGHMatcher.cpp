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
#include <set>
#include "BGHMatcher.h"
#include "opencv2/highgui.hpp"
#include <iostream>



namespace BGHMatcher
{
    // custom comparison operator for cv::Point
    // it can can be used to sort points by X then by Y
    struct cmpCvPoint {
        bool operator()(const cv::Point& a, const cv::Point& b) const {
            return (a.x < b.x) || ((a.x == b.x) && (a.y < b.y));
        }
    };

    
    void blur_img(cv::Mat& rsrc, cv::Mat& rdst, const int kblur, const int blur_type)
    {
        // it may not make much difference
        // but the border type can be changed here
        const int kborder = cv::BORDER_DEFAULT;
        switch (blur_type)
        {
        case BGHMatcher::BLUR_GAUSS:
            GaussianBlur(rsrc, rdst, { kblur, kblur }, 0, 0, kborder);
            break;
        case BGHMatcher::BLUR_MEDIAN:
            medianBlur(rsrc, rdst, kblur);
            break;
        case BGHMatcher::BLUR_BOX:
        default:
            blur(rsrc, rdst, { kblur, kblur }, { -1, -1 }, kborder);
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
        const double scale,
        BGHMatcher::T_ghough_table& rtable)
    {
        // sanity check scale value
        double fac = scale;
        if (fac < 0.1) fac = 0.1;
        if (fac > 10.0) fac = 10.0;

        // calculate centering offset
        int row_offset = rbgrad.rows / 2;
        int col_offset = rbgrad.cols / 2;

        // iterate through the binary gradient image pixel-by-pixel
        // use STL structures to build a lookup table dynamically
        std::map<uint8_t, std::map<cv::Point, uint16_t, cmpCvPoint>> lookup_table;
        for (int i = 0; i < rbgrad.rows; i++)
        {
            const uint8_t * pix = rbgrad.ptr<uint8_t>(i);
            for (int j = 0; j < rbgrad.cols; j++)
            {
                // get the "binary gradient" pixel value (key)
                const uint8_t uu = pix[j];
                if (rflags.get(uu))
                {
                    // the binary gradient value is in the set so add a vote
                    // the scaling operation can make one point have multiple votes
                    // so the vote count is mapped to a point and incremented
                    cv::Point offset_pt = cv::Point(col_offset - j, row_offset - i);
                    offset_pt.x = static_cast<int>(fac * offset_pt.x);
                    offset_pt.y = static_cast<int>(fac * offset_pt.y);
                    lookup_table[uu][offset_pt]++;
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
            rtable.elem[key].pt_votes = new T_pt_votes[n];
            size_t k = 0;
            for (const auto& rr : r.second)
            {
                cv::Point pt = rr.first;
                rtable.elem[key].pt_votes[k++] = { pt, rr.second };
                rtable.total_votes += rr.second;
                rtable.total_entries++;
            }
        }
    }

    void init_ghough_table_from_img(
        cv::Mat& rimg,
        BGHMatcher::T_ghough_table& rtable,
        const BGHMatcher::T_ghough_params& rparams)
    {
        cv::Mat img_bgrad;
        cv::Mat img_target;
        cv::Mat temp_mask;

        BGHMatcher::T_256_flags flags;

#if 1
        BGHMatcher::create_adjacent_bits_set(flags, BGHMatcher::N8_4ADJ);
#else
        // test for using every binary gradient pattern except 0
        flags.set_all();
        flags.clr(0);
#endif

        // create pre-blurred version of target image
        // same blur should be applied to input image when looking for matches
        blur_img(rimg, img_target, rparams.kblur, rparams.blur_type);

        // create binary gradient image from blurred target image
        BGHMatcher::cmp8NeighborsGT<uint8_t>(img_target, img_bgrad);

        // apply optional magnitude threshold mask to binary gradient image
        // a threshold >= 1.0 means 100% of points are used and no masking is done
        if ((rparams.mag_thr > 0.0) && (rparams.mag_thr < 1.0))
        {
            double qmax;
            cv::Mat temp_dx;
            cv::Mat temp_dy;
            cv::Mat temp_m;
            cv::Mat temp_a;
            const int SOBEL_DEPTH = CV_32F;

            // calculate X and Y gradients
            // they will become the gradient template images
            cv::Sobel(img_target, temp_dx, SOBEL_DEPTH, 1, 0, rparams.kblur);
            cv::Sobel(img_target, temp_dy, SOBEL_DEPTH, 0, 1, rparams.kblur);

            // create gradient magnitude mask
            // everything above the threshold (a fraction of the max) will be considered valid
            cv::cartToPolar(temp_dx, temp_dy, temp_m, temp_a);
            cv::minMaxLoc(temp_m, nullptr, &qmax);
            temp_mask = (temp_m > (qmax * rparams.mag_thr));
            
            // apply mask to binary gradient image
            img_bgrad = temp_mask & img_bgrad;
        }

        // create Generalized Hough lookup table from binary gradient image
        BGHMatcher::create_ghough_table(img_bgrad, flags, rparams.scale, rtable);
#if 0
        imshow("BG", img_bgrad);
#endif

        // save metadata for lookup table
        rtable.params = rparams;
    }
}
