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

#include <iostream>
#include <string>
#include "Knobs.h"


Knobs::Knobs() :
    is_op_required(false),
    is_equ_hist_enabled(false),
    is_record_enabled(false),
    kpreblur(7),
    kcliplimit(4),
    nchannel(Knobs::ALL_CHANNELS),
    noutmode(Knobs::OUT_COLOR),
    op_id(Knobs::OP_NONE),
    nimgscale(3),
    nksize(4),
    vimgscale({ 0.25, 0.325, 0.4, 0.5, 0.625, 0.75, 1.0 }),
    vksize({ -1, 1, 3, 5, 7})
{
}


Knobs::~Knobs()
{
}


void Knobs::show_help(void) const
{
    std::cout << std::endl;
    std::cout << "KEYS      FUNCTION" << std::endl;
    std::cout << "-----     ------------------------------------------------------" << std::endl;
    std::cout << "Esc       Quit" << std::endl;
    std::cout << "1,2,3,4   Choose BGR channel (Blue, Green, Red, BGR-to-Gray)" << std::endl;
    std::cout << "7,8,9,0   Output mode (raw match, gradients, pre-proc, color)" << std::endl;
    std::cout << "- or =    Adjust pre-blur (decrease, increase)" << std::endl;
    std::cout << "_ or +    Adjust CLAHE clip limit (decrease, increase)" << std::endl;
    std::cout << "[ or ]    Adjust image scale (decrease, increase)" << std::endl;
    std::cout << "{ or }    Adjust Sobel kernel size (decrease, increase)" << std::endl;
    std::cout << "e         Toggle histogram equalization" << std::endl;
    std::cout << "r         Toggle recording mode" << std::endl;
    std::cout << "t         Select next template from collection" << std::endl;
    std::cout << "u         Update Hough parameters from current settings" << std::endl;
    std::cout << "v         Create video from files in movie folder" << std::endl;
    std::cout << "?         Display this help info" << std::endl;
    std::cout << std::endl;
}


void Knobs::handle_keypress(const char ckey)
{
    bool is_valid = true;
    
    is_op_required = false;
    
    switch (ckey)
    {
        case '1':
        case '2':
        case '3':
        case '4':
        {
            // convert to channel code 0,1,2,3
            set_channel(ckey - '1');
            break;
        }
        case '7': set_output_mode(Knobs::OUT_RAW); break;
        case '8': set_output_mode(Knobs::OUT_GRAD); break;
        case '9': set_output_mode(Knobs::OUT_PREP); break;
        case '0': set_output_mode(Knobs::OUT_COLOR); break;
        case '+': inc_clip_limit(); break;
        case '_': dec_clip_limit(); break;
        case ']': inc_img_scale(); break;
        case '[': dec_img_scale(); break;
        case '=':
        {
            inc_pre_blur();
            break;
        }
        case '-':
        {
            dec_pre_blur();
            break;
        }
        case '}':
        {
            inc_ksize();
            is_op_required = true;
            op_id = Knobs::OP_UPDATE;
            break;
        }
        case '{':
        {
            dec_ksize();
            is_op_required = true;
            op_id = Knobs::OP_UPDATE;
            break;
        }
        case 'e':
        {
            toggle_equ_hist_enabled();
            break;
        }
        case 'r':
        {
            is_op_required = true;
            op_id = Knobs::OP_RECORD;
            toggle_record_enabled();
            break;
        }
        case 't':
        {
            is_op_required = true;
            op_id = Knobs::OP_TEMPLATE;
            break;
        }
        case 'u':
        {
            is_op_required = true;
            op_id = Knobs::OP_UPDATE;
            break;
        }
        case 'v':
        {
            is_op_required = true;
            op_id = Knobs::OP_MAKE_VIDEO;
            break;
        }
        case '?':
        {
            is_valid = false;
            show_help();
            break;
        }
        default:
        {
            is_valid = false;
            break;
        }
    }

    // display settings whenever valid keypress handled
    // except if it's an "op required" keypress
    if (is_valid && !is_op_required)
    {
        const std::vector<std::string> srgb({ "Blue ", "Green", "Red  ", "Gray " });
        const std::vector<std::string> sout({ "Raw  ", "Grad ", "Prep ", "Color" });
        std::cout << "Equ=" << is_equ_hist_enabled;
        std::cout << "  Clip=" << kcliplimit;
        std::cout << "  Ch=" << srgb[nchannel];
        std::cout << "  Blur=" << kpreblur;
        std::cout << "  Out=" << sout[noutmode];
        std::cout << "  Scale=" << vimgscale[nimgscale];
        std::cout << std::endl;
    }
}


bool Knobs::get_op_flag(int& ropid)
{
    bool result = is_op_required;
    ropid = op_id;
    is_op_required = false;
    return result;
}