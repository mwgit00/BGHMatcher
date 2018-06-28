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


    void create_ghough_table(
        const cv::Mat& rgrad,
        const double scale,
        BGHMatcher::T_ghough_table& rtable)
    {
        // sanity check scale value
        double fac = scale;
        if (fac < 0.1) fac = 0.1;
        if (fac > 10.0) fac = 10.0;

        // calculate centering offset
        int row_offset = rgrad.rows / 2;
        int col_offset = rgrad.cols / 2;

        // iterate through the gradient image pixel-by-pixel
        // use STL structures to build a lookup table dynamically
        std::map<uint8_t, std::map<cv::Point, uint16_t, cmpCvPoint>> lookup_table;
        for (int i = 0; i < rgrad.rows; i++)
        {
            const uint8_t * pix = rgrad.ptr<uint8_t>(i);
            for (int j = 0; j < rgrad.cols; j++)
            {
                // get the gradient pixel value (key)
                // everything non-zero is valid
                const uint8_t uu = pix[j];
                if (uu)
                {
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
        rtable.sz = rgrad.size();
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
        temp_ang.convertTo(rmgo, CV_8U, ang_step / (CV_2PI), 1.0);

        // apply mask to eliminate pixels
        rmgo &= temp_mask;
    }


    void init_ghough_table_from_img(
        cv::Mat& rimg,
        BGHMatcher::T_ghough_table& rtable,
        const BGHMatcher::T_ghough_params& rparams)
    {
        cv::Mat img_cgrad;
        create_masked_gradient_orientation_img(rimg, img_cgrad, rparams);

        cv::Mat img_target;
        GaussianBlur(rimg, img_target, { rparams.kblur, rparams.kblur }, 0);

        // create Generalized Hough lookup table from masked gradient image
        BGHMatcher::create_ghough_table(img_cgrad, rparams.scale, rtable);

#if 1
        cv::Mat img_display;
        normalize(img_cgrad, img_display, 0, 255, cv::NORM_MINMAX);
        imshow("GHTemplate", img_display);
#endif

        // save metadata for lookup table
        rtable.params = rparams;
    }

}
