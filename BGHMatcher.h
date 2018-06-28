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

#ifndef BGH_MATCHER_H_
#define BGH_MATCHER_H_

#include "opencv2/imgproc.hpp"


namespace BGHMatcher
{
    constexpr double ANG_STEP_MAX = 254.0;
    constexpr double ANG_STEP_MIN = 4.0;


    // structure that combines an image point and a vote count for that point
    typedef struct _T_pt_votes_struct
    {
        cv::Point pt;
        uint16_t votes;
        _T_pt_votes_struct() : pt{}, votes(0) {}
        _T_pt_votes_struct(const cv::Point& _pt, const uint16_t _v) : pt{_pt}, votes(_v) {}
    } T_pt_votes;
    

    // parameters used to create Generalized Hough lookup table
    typedef struct _T_ghough_params_struct
    {
        int kblur;
        int ksobel;
        double scale;
        double mag_thr;
        double ang_step;
        _T_ghough_params_struct() :
            kblur(7), ksobel(7), scale(1.0), mag_thr(1.0), ang_step(8.0) {}
        _T_ghough_params_struct(const int kb, const int ks, const double s, const double m, const double a) :
            kblur(kb), ksobel(ks), scale(s), mag_thr(m), ang_step(a) {}
    } T_ghough_params;
    
    
    // Non-STL data structure for Generalized Hough lookup table
    typedef struct _T_ghough_table_struct
    {
        T_ghough_params params;
        cv::Size sz;
        size_t total_votes;
        size_t total_entries;
        struct _elem_struct
        {
            size_t ct;
            T_pt_votes * pt_votes;
            _elem_struct() : ct(0), pt_votes(nullptr) {}
            void clear() { ct = 0;  if (pt_votes) { delete[] pt_votes; } pt_votes = nullptr; }
        } elem[256];
        _T_ghough_table_struct() : params(), sz(0, 0), total_votes(0), total_entries(0) {}
        void clear()
        {
            sz = { 0,0 };
            total_votes = 0;
            total_entries = 0;
            for (size_t i = 0; i < 256; i++) { elem[i].clear(); }
        }
    } T_ghough_table;


    // Applies Generalized Hough transform to an encoded gradient image (CV_8U).
    // The size of the target image used to generate the table will constrain the results.
    // Pixels near border and within half the X or Y dimensions of target image will be 0.
    // Template parameters specify output type.  Try <CV_32F,float> or <CV_16U,uint16_t>.
    // Output image is same size as input.  Maxima indicate good matches.
    template<int E, typename T>
    void apply_ghough_transform(
        const cv::Mat& rimg,
        cv::Mat& rout,
        const BGHMatcher::T_ghough_table& rtable)
    {
        rout = cv::Mat::zeros(rimg.size(), E);
        for (int i = rtable.sz.height / 2; i < rimg.rows - rtable.sz.height / 2; i++)
        {
            const uint8_t * pix = rimg.ptr<uint8_t>(i);
            for (int j = rtable.sz.width / 2; j < rimg.cols - rtable.sz.width / 2; j++)
            {
                // look up voting table for pixel
                // iterate through the points (if any) and add votes
                uint8_t uu = pix[j];
                T_pt_votes * pt_votes = rtable.elem[uu].pt_votes;
                const size_t ct = rtable.elem[uu].ct;
                for (size_t k = 0; k < ct; k++)
                {
                    const cv::Point& rp = pt_votes[k].pt;
                    int mx = (j + rp.x);
                    int my = (i + rp.y);
                    T * pix = rout.ptr<T>(my) + mx;
                    *pix += pt_votes[k].votes;
                }
            }
        }
    }


    // Applies Generalized Hough transform to an input encoded gradient image (CV_8U).
    // Each vote is range-checked.  Votes that would fall outside the image are discarded.
    // Template parameters specify output type.  Try <CV_32F,float> or <CV_16U,uint16_t>.
    // Output image is same size as input.  Maxima indicate good matches.
    template<int E, typename T>
    void apply_ghough_transform_allpix(
        const cv::Mat& rimg,
        cv::Mat& rout,
        const BGHMatcher::T_ghough_table& rtable)
    {
        rout = cv::Mat::zeros(rimg.size(), E);
        for (int i = 1; i < (rimg.rows - 1); i++)
        {
            const uint8_t * pix = rimg.ptr<uint8_t>(i);
            for (int j = 1; j < (rimg.cols - 1); j++)
            {
                // look up voting table for pixel
                // iterate through the points (if any) and add votes
                uint8_t uu = pix[j];
                T_pt_votes * pt_votes = rtable.elem[uu].pt_votes;
                const size_t ct = rtable.elem[uu].ct;
                for (size_t k = 0; k < ct; k++)
                {
                    // only vote if pixel is within output image bounds
                    const cv::Point& rp = pt_votes[k].pt;
                    int mx = (j + rp.x);
                    int my = (i + rp.y);
                    if ((mx >= 0) && (mx < rout.cols) &&
                        (my >= 0) && (my < rout.rows))
                    {
                        T * pix = rout.ptr<T>(my) + mx;
                        *pix += pt_votes[k].votes;
                    }
                }
            }
        }
    }


    // This is the preprocessing step for the "classic" Generalized Hough algorithm.
    // Calculates Sobel derivatives of input grayscale image.  Converts to polar coordinates and
    // finds magnitude and angle (orientation).  Converts angle to integer with 4 to 254 steps.
    // Masks the pixels with gradient magnitudes above a threshold.
    void create_masked_gradient_orientation_img(
        const cv::Mat& rimg,
        cv::Mat& rmgo,
        const BGHMatcher::T_ghough_params& rparams);

    
    // Creates a Generalized Hough lookup table from encoded gradient input image (CV_8U).
    // The scale parameter shrinks or expands the point set.
    void create_ghough_table(
        const cv::Mat& rgrad,
        const double scale,
        BGHMatcher::T_ghough_table& rtable);


    // Helper function for initializing Generalized Hough table from grayscale image.
    // Default parameters are good starting point for doing object identification.
    // Table must be a newly created object with blank data.
    void init_ghough_table_from_img(
        cv::Mat& rimg,
        BGHMatcher::T_ghough_table& rtable,
        const BGHMatcher::T_ghough_params& rparams);
}

#endif // BGH_MATCHER_H_
