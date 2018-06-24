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


// Bit index and variable naming convention for 8-neighbor masks:
//
// 5 6 7      nn n0 np
// 4 * 0      0n 00 0p
// 3 2 1      pn p0 pp


namespace BGHMatcher
{
    constexpr double ANG_STEP_MAX = 254.0;
    constexpr double ANG_STEP_MIN = 4.0;
    constexpr double RNG_FAC = 255.0;


    // Masks for selecting number of adjacent bits to consider.
    // Using 4 adjacent bits seems to be best default choice.
    enum
    {
        N8_3ADJ = (1 << 2), // consider 3 adjacent bits (might help or hurt)
        N8_4ADJ = (1 << 3), // consider 4 adjacent bits (usually best choice by itself)
        N8_5ADJ = (1 << 4), // consider 5 adjacent bits (might help or hurt)
        N8_3OR4 = (3 << 2), // consider 3 and 4 adjacent bits
        N8_4OR5 = (3 << 3), // consider 4 and 5 adjacent bits
        N8_345 =  (7 << 2), // consider 3, 4, and 5 adjacent bits
    };
    

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


    // Membership "set" for values 0-255 implemented as 256 bit flags.
    typedef struct _T_256_flags_struct
    {
        uint32_t bits[8];
        _T_256_flags_struct() { memset(bits, 0, sizeof(bits)); }
        ~_T_256_flags_struct() {}
        void set(const uint8_t n) { bits[n >> 5] |= (1 << (n & 0x1F)); }
        void clr(const uint8_t n) { bits[n >> 5] &= ~(1 << (n & 0x1F)); }
        bool get(const uint8_t n) const { return (bits[n >> 5] & (1 << (n & 0x1F))) ? true : false; }
        void set_all(void) { memset(bits, 0xFF, sizeof(bits)); }
    } T_256_flags;


    // This produces a "binary gradient" image with features for the Generalized Hough transform.
    // It compares a central pixel with its 8-neighbors and sets bits if central pixel is larger.
    // An output pixel with a value of 255 is greater than all of its 8-neighbors.  The range
    // (max-min) of the 3x3 neighborhood can be used as a threshold to mask pixels at strong
    // gradients.  Pixels on weak gradients are set to 0.  A threshold of 0 masks NO pixels.
    // Input image is grayscale.  Template parameter specifies input pixel type, usually uint8_t.
    // Output image is CV_8U type with same size as input image.  Border pixels are 0.
    template<typename T>
    void cmp8NeighborsGTRng(const cv::Mat& rsrc, cv::Mat& rdst, const uint8_t rng = 0)
    {
        // output is always 8-bit unsigned
        rdst = cv::Mat::zeros(rsrc.size(), CV_8U);
        for (int i = 1; i < rsrc.rows - 1; i++)
        {
            // initialize pointers to central pixel and its 8-neighbors
            const T * pixsnn = rsrc.ptr<T>(i - 1);
            const T * pixs0n = rsrc.ptr<T>(i + 0);
            const T * pixspn = rsrc.ptr<T>(i + 1);
            const T * pixsn0 = pixsnn + 1;
            const T * pixs00 = pixs0n + 1;
            const T * pixsp0 = pixspn + 1;
            const T * pixsnp = pixsn0 + 1;
            const T * pixs0p = pixs00 + 1;
            const T * pixspp = pixsp0 + 1;

            // initialize pointer to output pixel
            uint8_t * pixd = rdst.ptr<uint8_t>(i) + 1;

            // iterate along row
            // if the max of the 3x3 neighborhood exceeds the min by a threshold
            // then the new output pixel value is the "greater than" mask for the 8-neighbors
            for (int j = 1; j < rsrc.cols - 1; j++)
            {
                T q = *(pixs00);
                T umin = q;
                T umax = q;
                uint8_t bg = 0;

                // if range threshold is not zero then find min-max of 3x3 neighborhood
                if (rng > 0)
                {
                    // determine maximum of center pixel and 8-neighbors
                    umax = cv::max(umax, *(pixs0p));
                    umax = cv::max(umax, *(pixspp));
                    umax = cv::max(umax, *(pixsp0));
                    umax = cv::max(umax, *(pixspn));
                    umax = cv::max(umax, *(pixs0n));
                    umax = cv::max(umax, *(pixsnn));
                    umax = cv::max(umax, *(pixsn0));
                    umax = cv::max(umax, *(pixsnp));

                    // determine minimum of center pixel and 8-neighbors
                    umin = cv::min(umin, *(pixs0p));
                    umin = cv::min(umin, *(pixspp));
                    umin = cv::min(umin, *(pixsp0));
                    umin = cv::min(umin, *(pixspn));
                    umin = cv::min(umin, *(pixs0n));
                    umin = cv::min(umin, *(pixsnn));
                    umin = cv::min(umin, *(pixsn0));
                    umin = cv::min(umin, *(pixsnp));
                }

                // if max-min exceeds the threshold or is 0
                // then do "greater-than" encoding for 8-neighbors
                if ((umax - umin) >= rng)
                {
                    bg =
                        ((q > *(pixs0p))     ) |
                        ((q > *(pixspp)) << 1) |
                        ((q > *(pixsp0)) << 2) |
                        ((q > *(pixspn)) << 3) |
                        ((q > *(pixs0n)) << 4) |
                        ((q > *(pixsnn)) << 5) |
                        ((q > *(pixsn0)) << 6) |
                        ((q > *(pixsnp)) << 7);
                }

                // scoot the pointers for the 3x3 neighborhood
                pixs0p++;
                pixspp++;
                pixsp0++;
                pixspn++;
                pixs0n++;
                pixsnn++;
                pixsn0++;
                pixsnp++;
                pixs00++;

                // set the output pixel and scoot its pointer
                *(pixd++) = bg;
            }
        }
    }


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


    // Creates a set with all 8-bit values that have the specified number(s) of adjacent bits.
    // The bits in the mask specify the number of adjacent bits to consider, 1 through 8.
    void create_adjacent_bits_set(
        BGHMatcher::T_256_flags& rflags,
        const uint8_t mask);


    // Calculates gradient magnitude of input grayscale image.  Creates a mask for all
    // gradient magnitudes above a threshold.  Applies the mask to an input/output image.
    // Pixels at small gradients will be set to 0 in the input/output CV_8U image.
    void apply_sobel_gradient_mask(
        const cv::Mat& rimg,
        cv::Mat& rmod,
        const int kblur,
        const double mag_thr);


    // This is the preprocessing step for the "classic" Generalized Hough algorithm.
    // Calculates Sobel derivatives of input grayscale image.  Converts to polar coordinates and
    // finds magnitude and angle (orientation).  Converts angle to integer with 4 to 254 steps.
    // Masks the pixels with gradient magnitudes above a threshold.
    void create_masked_gradient_orientation_img(
        const cv::Mat& rimg,
        cv::Mat& rmgo,
        const BGHMatcher::T_ghough_params& rparams);

    
    // Creates a Generalized Hough lookup table from encoded gradient input image (CV_8U).
    // A set of flags determines which encoded gradient pixel values to use.
    // The scale parameter shrinks or expands the point set.
    void create_ghough_table(
        const cv::Mat& rbgrad,
        const BGHMatcher::T_256_flags& rflags,
        const double scale,
        BGHMatcher::T_ghough_table& rtable);


    void init_binary_ghough_table_from_img(
        cv::Mat& rimg,
        BGHMatcher::T_ghough_table& rtable,
        const BGHMatcher::T_ghough_params& rparams);


    // Helper function for initializing Generalized Hough table from grayscale image.
    // This uses a hybrid approach with "binary gradients" and Sobel magnitude mask.
    // Default parameters are good starting point for doing object identification.
    // Table must be a newly created object with blank data.
    void init_hybrid_ghough_table_from_img(
        cv::Mat& rimg,
        BGHMatcher::T_ghough_table& rtable,
        const BGHMatcher::T_ghough_params& rparams);


    // Helper function for initializing Generalized Hough table from grayscale image.
    // This uses the classic approach of encoding the Sobel gradient orientation.
    // Default parameters are good starting point for doing object identification.
    // Table must be a newly created object with blank data.
    void init_classic_ghough_table_from_img(
        cv::Mat& rimg,
        BGHMatcher::T_ghough_table& rtable,
        const BGHMatcher::T_ghough_params& rparams);
}

#endif // BGH_MATCHER_H_
