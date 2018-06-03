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
    // Masks for selecting number of adjacent bits to consider.
    // Using combinations of 4 or 5 adjacent pixels seems to be best choice.
    enum
    {
        N8_4ADJ = (1 << 3), // consider 4 adjacent pixels (usually best choice)
        N8_5ADJ = (1 << 4), // consider 5 adjacent pixels
        N8_4OR5 = (3 << 3), // consider 4 or 5 adjacent pixels
    };
    
    // Non-STL data structure for Generalized Hough data "template"
    typedef struct _T_ghough_data_struct
    {
    public:
        cv::Size sz;
        int kblur;
        size_t total;
        struct _elem_struct
        {
            size_t ct;
            cv::Point * pts;
            _elem_struct() : ct(0), pts(nullptr) {}
            ~_elem_struct() { if (pts != nullptr) { delete[] pts; } }
        } elem[256];
        _T_ghough_data_struct() : sz(0, 0), kblur(0), total(0) {}
    } T_ghough_data;


    // Membership "set" for values 0-255 implemented as 256 bit flags.
    typedef struct _T_256_flags_struct
    {
    private:
        uint32_t bits[8];
    public:
        _T_256_flags_struct() { memset(bits, 0, sizeof(bits)); }
        ~_T_256_flags_struct() {}
        void set(const uint8_t n) { bits[n >> 5] |= (1 << (n & 0x1F)); }
        void clr(const uint8_t n) { bits[n >> 5] &= ~(1 << (n & 0x1F)); }
        bool get(const uint8_t n) const { return (bits[n >> 5] & (1 << (n & 0x1F))) ? true : false; }
    } T_256_flags;

    
    // Compares a central pixel with its 8-neighbors and sets bits if central pixel is larger.
    // This produces a "binary gradient" image with features for Generalized Hough transform.
    // An output pixel with a value of 255 is greater than all of its 8-neighbors.
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
            T * pixd = rdst.ptr<T>(i) + 1;

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


    // Compares a central pixel with its 8-neighbors and sets bits if central pixel is smaller.
    // This produces a "binary gradient" image with features for Generalized Hough transform.
    // An output pixel with a value of 255 is less than all of its 8-neighbors.
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
            T * pixd = rdst.ptr<T>(i) + 1;

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


    // Creates a set with all 8-bit values that have the specified number(s) of adjacent bits.
    // The bits in the mask specify the number of adjacent bits to consider.
    void create_adjacent_bits_set(
        BGHMatcher::T_256_flags& rflags,
        const uint8_t mask = BGHMatcher::N8_4ADJ);


    // Creates a Generalized Hough data "template" from binary gradient input image (CV_8U).
    // A set of flags determines which binary gradient pixel values to use.
    void create_ghough_data(
        const cv::Mat& rbgrad,
        const BGHMatcher::T_256_flags& rflags,
        BGHMatcher::T_ghough_data& rdata);


    // Applies Generalized Hough data "template" to an input image.
    // Output is CV_8U image of same size as input.  Maxima indicate good matches.
    // Pixels near border and within half the X or Y dimensions of the template will be 0.
    void apply_ghough_transform(
        const cv::Mat& rimg,
        cv::Mat& rout,
        const BGHMatcher::T_ghough_data& rdata);
};

#endif // BGH_MATCHER_H_
