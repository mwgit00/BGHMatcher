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

#ifndef KNOBS_H_
#define KNOBS_H_

#include <vector>

class Knobs
{
public:

    enum
    {
        ALL_CHANNELS = 3,
    };

    enum
    {
        OUT_RAW = 0,
        OUT_MASK,
        OUT_COLOR,
    };

    enum
    {
        OP_NONE = 0,
        OP_TEMPLATE,
        OP_UPDATE,
        OP_RECORD,
        OP_MAKE_VIDEO,
    };

    Knobs();
    virtual ~Knobs();

    void show_help(void) const;

    bool get_op_flag(int& ropid);

    bool get_equ_hist_enabled(void) const { return is_equ_hist_enabled; }
    void toggle_equ_hist_enabled(void) { is_equ_hist_enabled = !is_equ_hist_enabled; }

    bool get_record_enabled(void) const { return is_record_enabled; }
    void toggle_record_enabled(void) { is_record_enabled = !is_record_enabled; }

    int get_pre_blur(void) const { return kpreblur; }
    void inc_pre_blur(void) { kpreblur = (kpreblur < 35) ? kpreblur + 2 : kpreblur; }
    void dec_pre_blur(void) { kpreblur = (kpreblur > 1) ? kpreblur - 2 : kpreblur; };

    int get_clip_limit(void) const { return kcliplimit; }
    void inc_clip_limit(void) { kcliplimit = (kcliplimit < 20) ? kcliplimit + 1 : kcliplimit; }
    void dec_clip_limit(void) { kcliplimit = (kcliplimit > 0) ? kcliplimit - 1 : kcliplimit; };

    int get_channel(void) const { return nchannel; }
    void set_channel(const int n) { nchannel = n; }

    int get_output_mode(void) const { return noutmode; }
    void set_output_mode(const int n) { noutmode = n; }

    double get_img_scale(void) const { return vimgscale[nimgscale]; }
    void inc_img_scale(void) { nimgscale = (nimgscale < (vimgscale.size() - 1)) ? nimgscale + 1 : nimgscale; }
    void dec_img_scale(void) { nimgscale = (nimgscale > 0) ? nimgscale - 1 : nimgscale; };

    double get_ksize(void) const { return vksize[nksize]; }
    void inc_ksize(void) { nksize = (nksize < (vksize.size() - 1)) ? nksize + 1 : nksize; }
    void dec_ksize(void) { nksize = (nksize > 0) ? nksize - 1 : nksize; };

    void handle_keypress(const char c);

private:

    // One-shot flag for signaling when extra operation needs to be done
    // before continuing image processing loop
    bool is_op_required;

    // Flag for enabling histogram equalization
    bool is_equ_hist_enabled;

    // Flag for enabling recording
    bool is_record_enabled;

    // Amount of Gaussian blurring in preprocessing step
    int kpreblur;

    // Clip limit for CLAHE
    int kcliplimit;

    // Channel selection (B,G,R, or Gray)
    int nchannel;

    // Output mode (raw, mask, or color)
    int noutmode;

    // Type of operation that is required
    int op_id;

    // Index of currently selected scale factor
    size_t nimgscale;

    // Index of currently selected Sobel kernel size
    size_t nksize;

    // Array of supported scale factors
    std::vector<double> vimgscale;

    // Array of supported Sobel kernel sizes
    std::vector<int> vksize;
};

#endif // KNOBS_H_
