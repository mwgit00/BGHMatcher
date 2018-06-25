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


namespace BGHMatcher
{
    // custom comparison operator for cv::Point
    // it can can be used to sort points by X then by Y
    struct cmpCvPoint {
        bool operator()(const cv::Point& a, const cv::Point& b) const {
            return (a.x < b.x) || ((a.x == b.x) && (a.y < b.y));
        }
    };


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


    void apply_sobel_gradient_mask(
        const cv::Mat& rimg,
        cv::Mat& rmod,
        const int kblur,
        const double mag_thr)
    {
        // proceed if threshold is in range 0-1
        // 0 will not mask out any pixels, 1 will mask out all pixels
        if ((mag_thr >= 0.0) && (mag_thr < 1.0))
        {
            double qmax;
            cv::Mat temp_dx;
            cv::Mat temp_dy;
            cv::Mat temp_mag;
            cv::Mat temp_a;
            cv::Mat temp_mask;
            const int SOBEL_DEPTH = CV_32F;

            // calculate X and Y gradients for input image
            cv::Sobel(rimg, temp_dx, SOBEL_DEPTH, 1, 0, kblur);
            cv::Sobel(rimg, temp_dy, SOBEL_DEPTH, 0, 1, kblur);

            // create gradient magnitude mask
            // everything above a fraction of the max will be kept
            cv::cartToPolar(temp_dx, temp_dy, temp_mag, temp_a);
            cv::minMaxLoc(temp_mag, nullptr, &qmax);
            temp_mask = (temp_mag > (qmax * mag_thr));

            // apply mask to image to be modified
            rmod = temp_mask & rmod;
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

        // iterate through the gradient image pixel-by-pixel
        // use STL structures to build a lookup table dynamically
        std::map<uint8_t, std::map<cv::Point, uint16_t, cmpCvPoint>> lookup_table;
        for (int i = 0; i < rbgrad.rows; i++)
        {
            const uint8_t * pix = rbgrad.ptr<uint8_t>(i);
            for (int j = 0; j < rbgrad.cols; j++)
            {
                // get the gradient pixel value (key)
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

        // blow away any old data in table
        rtable.clear();

        // then put lookup table into a fixed non-STL structure
        // that is much more efficient when running debug code
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

    


    void create_masked_gradient_orientation_img(
        const cv::Mat& rimg,
        cv::Mat& rmgo,
        const BGHMatcher::T_ghough_params& rparams)
    {
        double qmax;
        double qmin;
        double ang_step = rparams.ang_step;
        cv::Mat temp_dx;
        cv::Mat temp_dy;
        cv::Mat temp_mag;
        cv::Mat temp_ang;
        cv::Mat temp_mask;
        const int SOBEL_DEPTH = CV_32F;

        // calculate X and Y gradients for input image
        cv::Sobel(rimg, temp_dx, SOBEL_DEPTH, 1, 0, rparams.ksobel);
        cv::Sobel(rimg, temp_dy, SOBEL_DEPTH, 0, 1, rparams.ksobel);

        // convert X-Y gradients to magnitude and angle
        cartToPolar(temp_dx, temp_dy, temp_mag, temp_ang);

        // create mask for pixels that exceed gradient magnitude threshold
        minMaxLoc(temp_mag, nullptr, &qmax);
        temp_mask = (temp_mag > (qmax * rparams.mag_thr));

        // scale, offset, and convert the angle image so 0-2pi becomes integers 1 to (ANG_STEP+1)
        ang_step = (ang_step > ANG_STEP_MAX) ? ANG_STEP_MAX : ang_step;
        ang_step = (ang_step < ANG_STEP_MIN) ? ANG_STEP_MIN: ang_step;
        temp_ang.convertTo(rmgo, CV_8U, ang_step / (CV_2PI), 2.0);

        // apply mask to eliminate pixels
        rmgo &= temp_mask;
    }


    void init_binary_ghough_table_from_img(
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
        GaussianBlur(rimg, img_target, { rparams.kblur, rparams.kblur }, 0);

        // create encoded gradient image from blurred target image
        int krng = static_cast<int>(rparams.mag_thr * RNG_FAC);
        BGHMatcher::cmp8NeighborsGTRng<uint8_t>(img_target, img_bgrad, krng);

        // create Generalized Hough lookup table from masked binary gradient image
        BGHMatcher::create_ghough_table(img_bgrad, flags, rparams.scale, rtable);

#if 1
        imshow("GHTemplate", img_bgrad);
#endif

        // save metadata for lookup table
        rtable.params = rparams;
    }



    void init_hybrid_ghough_table_from_img(
        cv::Mat& rimg,
        BGHMatcher::T_ghough_table& rtable,
        const BGHMatcher::T_ghough_params& rparams)
    {
        cv::Mat img_bgrad;
        cv::Mat img_target;
        cv::Mat temp_mask;

        BGHMatcher::T_256_flags flags;

#if 0
        BGHMatcher::create_adjacent_bits_set(flags, BGHMatcher::N8_4ADJ);
#else
        // test for using every binary gradient pattern except 0
        flags.set_all();
        flags.clr(0);
#endif

        // create pre-blurred version of target image
        // same blur should be applied to input image when looking for matches
        GaussianBlur(rimg, img_target, { rparams.kblur, rparams.kblur }, 0);

        // create binary gradient image from blurred target image
        // and apply magnitude threshold mask to binary gradient image
        BGHMatcher::cmp8NeighborsGTRng<uint8_t>(img_target, img_bgrad);
        apply_sobel_gradient_mask(rimg, img_bgrad, rparams.ksobel, rparams.mag_thr);

        // create Generalized Hough lookup table from masked binary gradient image
        BGHMatcher::create_ghough_table(img_bgrad, flags, rparams.scale, rtable);

#if 1
        imshow("GHTemplate", img_bgrad);
#endif

        // save metadata for lookup table
        rtable.params = rparams;
    }


    void init_classic_ghough_table_from_img(
        cv::Mat& rimg,
        BGHMatcher::T_ghough_table& rtable,
        const BGHMatcher::T_ghough_params& rparams)
    {
        cv::Mat img_cgrad;
        create_masked_gradient_orientation_img(rimg, img_cgrad, rparams);

        cv::Mat img_target;
        GaussianBlur(rimg, img_target, { rparams.kblur, rparams.kblur }, 0);

        BGHMatcher::T_256_flags flags;
        flags.set_all();
        flags.clr(0);
        
        // create Generalized Hough lookup table from masked classic gradient image
        BGHMatcher::create_ghough_table(img_cgrad, flags, rparams.scale, rtable);

#if 1
        cv::Mat img_display;
        normalize(img_cgrad, img_display, 0, 255, cv::NORM_MINMAX);
        imshow("GHTemplate", img_display);
#endif

        // save metadata for lookup table
        rtable.params = rparams;
    }

}
