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
    enum
    {
        BLUR_BOX,
        BLUR_GAUSS,
        BLUR_MEDIAN,
    };

    // Masks for selecting number of adjacent bits to consider.
    // Using combinations of 4 or 5 adjacent pixels seems to be best choice.
    enum
    {
        N8_4ADJ = (1 << 3), // consider 4 adjacent pixels (usually best choice)
        N8_5ADJ = (1 << 4), // consider 5 adjacent pixels
        N8_4OR5 = (3 << 3), // consider 4 or 5 adjacent pixels
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
        int blur_type;
        double scale;
        double mag_thr;
        _T_ghough_params_struct() : kblur(7), blur_type(BLUR_GAUSS), scale(1.0), mag_thr(1.0) {}
        _T_ghough_params_struct(const int i, const int j, const double s, const double t) :
            kblur(i), blur_type(j), scale(s), mag_thr(t) {}
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


    // This produces a "binary gradient" image with features for Generalized Hough transform.
    // Input image is grayscale.  Template parameter specifies input pixel type, usually uint8_t.
    // Compares a central pixel with its 8-neighbors and sets bits if central pixel is larger.
    // An output pixel with a value of 255 is greater than all of its 8-neighbors.
    // Output image is CV_8U type with same size as input image.  Border pixels are 0.
    template<typename T>
    void cmp8NeighborsGT(const cv::Mat& rsrc, cv::Mat& rdst)
    {
        // output is always 8-bit unsigned
        rdst = cv::Mat::zeros(rsrc.size(), CV_8U);
        for (int i = 1; i < rsrc.rows - 1; i++)
        {
            // initialize pointers to central pixel's 8-neighbors
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
            // new output pixel value is the "greater than" mask for its 8-neighbors
            // increment pointers for the next 8-neighbors
            for (int j = 1; j < rsrc.cols - 1; j++)
            {
                T q = *(pixs00++);
                int b =
                    ((*(pixs0p++) > q)) |
                    ((*(pixspp++) > q) << 1) |
                    ((*(pixsp0++) > q) << 2) |
                    ((*(pixspn++) > q) << 3) |
                    ((*(pixs0n++) > q) << 4) |
                    ((*(pixsnn++) > q) << 5) |
                    ((*(pixsn0++) > q) << 6) |
                    ((*(pixsnp++) > q) << 7);
                *(pixd++) = static_cast<uint8_t>(b);
            }
        }
    }


    // This produces a "binary gradient" image with features for Generalized Hough transform.
    // Input image is grayscale.  Template parameter specifies input pixel type, usually uint8_t.
    // Compares a central pixel with its 8-neighbors and sets bits if central pixel is larger.
    // An output pixel with a value of 255 is less than all of its 8-neighbors.
    // Output image is CV_8U type with same size as input image.  Border pixels are 0.
    template<typename T>
    void cmp8NeighborsLT(const cv::Mat& rsrc, cv::Mat& rdst)
    {
        // output is always 8-bit unsigned
        rdst = cv::Mat::zeros(rsrc.size(), CV_8U);
        for (int i = 1; i < rsrc.rows - 1; i++)
        {
            // initialize pointers to central pixel's 8-neighbors
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
            uint8_t * pixd = rdst.ptr<T>(i) + 1;

            // iterate along row
            // new output pixel value is the "less than" mask for its 8-neighbors
            // increment pointers for the next 8-neighbors
            for (int j = 1; j < rsrc.cols - 1; j++)
            {
                T q = *(pixs00++);
                int b =
                    ((*(pixs0p++) < q)) |
                    ((*(pixspp++) < q) << 1) |
                    ((*(pixsp0++) < q) << 2) |
                    ((*(pixspn++) < q) << 3) |
                    ((*(pixs0n++) < q) << 4) |
                    ((*(pixsnn++) < q) << 5) |
                    ((*(pixsn0++) < q) << 6) |
                    ((*(pixsnp++) < q) << 7);
                *(pixd++) = static_cast<uint8_t>(b);
            }
        }
    }


    // Applies Generalized Hough transform to an input binary gradient image (CV_8U).
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


    // Applies Generalized Hough transform to an input binary gradient image (CV_8U).
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

    
    // Calculate gradient magnitude of input grayscale image.
    // Create a mask for all gradient magnitudes above a threshold.
    // Use mask to zero pixels in a second CV_8U image corresponding to small gradients.
    void apply_sobel_gradient_mask(
        const cv::Mat& rimg,
        cv::Mat& rmod,
        const int kblur,
        const double mag_thr);

    
    // Make new blurred image using specified kernel size and blurring type.
    // Blurring operation can be done in-place.
    void blur_img(
        cv::Mat& rsrc,
        cv::Mat& rdst,
        const int kblur,
        const int blur_type);


    // Creates a set with all 8-bit values that have the specified number(s) of adjacent bits.
    // The bits in the mask specify the number of adjacent bits to consider.
    // Default argument is for using 4 adjacent bits which seems like best starting point.
    void create_adjacent_bits_set(
        BGHMatcher::T_256_flags& rflags,
        const uint8_t mask = N8_4ADJ);


    // Creates a Generalized Hough lookup table from binary gradient input image (CV_8U).
    // A set of flags determines which binary gradient pixel values to use.
    // The scale parameter shrinks or expands the point set.
    void create_ghough_table(
        const cv::Mat& rbgrad,
        const BGHMatcher::T_256_flags& rflags,
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
