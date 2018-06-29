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

#include "Windows.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>

#include "BGHMatcher.h"
#include "Knobs.h"
#include "util.h"


#define MATCH_DISPLAY_THRESHOLD (0.8)           // arbitrary
#define MOVIE_PATH              ".\\movie\\"    // user may need to create or change this
#define DATA_PATH               ".\\data\\"     // user may need to change this


using namespace cv;


#define SCA_BLACK   (cv::Scalar(0,0,0))
#define SCA_RED     (cv::Scalar(0,0,255))
#define SCA_GREEN   (cv::Scalar(0,255,0))
#define SCA_BLUE    (cv::Scalar(255,0,0))
#define SCA_MAGENTA (cv::Scalar(255,0,255))
#define SCA_YELLOW  (cv::Scalar(0,255,255))
#define SCA_WHITE   (cv::Scalar(255,255,255))


Mat template_image;
const char * stitle = "BGHMatcher";
const double default_mag_thr = 0.1;
int n_record_ctr = 0;
size_t nfile = 0;

const std::vector<T_file_info> vfiles =
{
    { default_mag_thr, "circle_b_on_w.png" },
    { default_mag_thr, "ring_b_on_w.png" },
    { default_mag_thr, "bottle_20perc_top_b_on_w.png" },
    { default_mag_thr, "panda_face.png" },
    { default_mag_thr, "stars_main.png" }
};


bool wait_and_check_keys(Knobs& rknobs)
{
    bool result = true;

    int nkey = waitKey(1);
    char ckey = static_cast<char>(nkey);

    // check that a keypress has been returned
    if (nkey >= 0)
    {
        if (ckey == 27)
        {
            // done if ESC has been pressed
            result = false;
        }
        else
        {
            rknobs.handle_keypress(ckey);
        }
    }

    return result;
}


void image_output(
    Mat& rimg,
    const double qmax,
    const Point& rptmax,
    const Knobs& rknobs,
    BGHMatcher::T_ghough_table& rtable)
{
    const int h_score = 16;
    const double scale = rtable.params.scale;

    // determine size of "target" box
    // it will vary depending on the scale parameter
    Size rsz = rtable.img_sz;
    rsz.height *= scale;
    rsz.width *= scale;
    Point corner = { rptmax.x - rsz.width / 2, rptmax.y - rsz.height / 2 };

    // format score string for viewer (#.##)
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << (qmax / rtable.total_votes);

    // draw current template in upper right corner with blue box around it
    Mat bgr_template_img;
    cvtColor(template_image, bgr_template_img, COLOR_GRAY2BGR);
    Size osz = rimg.size();
    Size tsz = template_image.size();
    Rect roi = cv::Rect(osz.width - tsz.width, 0, tsz.width, tsz.height);
    bgr_template_img.copyTo(rimg(roi));

    // save each frame to a file if recording
    // and use magenta box around template image to indicate recording mode is active
    cv::Scalar box_color = SCA_BLUE;
    if (rknobs.get_record_enabled())
    {
        std::ostringstream osx;
        osx << MOVIE_PATH << "img_" << std::setfill('0') << std::setw(5) << n_record_ctr << ".png";
        imwrite(osx.str(), rimg);
        n_record_ctr++;
        box_color = SCA_MAGENTA;
    }

    rectangle(rimg, { osz.width - tsz.width, 0 }, { osz.width, tsz.height }, box_color, 2);

    // draw black background box then draw text score on top of it
    rectangle(rimg, { corner.x,corner.y - h_score, 40, h_score }, SCA_BLACK, -1);
    putText(rimg, oss.str(), { corner.x,corner.y - 4 }, FONT_HERSHEY_PLAIN, 1.0, SCA_WHITE, 1);

    // draw rectangle around best match with yellow dot at center
    rectangle(rimg, { corner.x, corner.y, rsz.width, rsz.height }, SCA_GREEN, 2);
    circle(rimg, rptmax, 2, SCA_YELLOW, -1);

    cv::imshow(stitle, rimg);
}


void reload_template(
    const Knobs& rknobs,
    BGHMatcher::T_ghough_table& rtable,
    const T_file_info& rinfo)
{
    int kblur = rknobs.get_pre_blur();
    int ksobel = rknobs.get_ksize();
    std::string spath = DATA_PATH + rinfo.sname;
    template_image = imread(spath, IMREAD_GRAYSCALE);
    
    BGHMatcher::init_ghough_table_from_img(
        template_image, rtable, { kblur, ksobel, 1.0, rinfo.mag_thr, 8.0 });
    
    std::cout << "Loaded template (blur,sobel) = " << kblur << "," << ksobel << "): ";
    std::cout << rinfo.sname << " " << rtable.total_votes << std::endl;
}


void loop(void)
{
    Knobs theKnobs;
    int op_id;

    double qmax;
    Size capture_size;
    Point ptmax;
    
    Mat img;
    Mat img_viewer;
    Mat img_gray;
    Mat img_grad;
    Mat img_channels[3];
    Mat img_match;

    BGHMatcher::T_ghough_table theGHData;
    Ptr<CLAHE> pCLAHE = createCLAHE();

    // need a 0 as argument
    VideoCapture vcap(0);
    if (!vcap.isOpened())
    {
        std::cout << "Failed to open VideoCapture device!" << std::endl;
        ///////
        return;
        ///////
    }

    // camera is ready so grab a first image to determine its full size
    vcap >> img;
    capture_size = img.size();

    // use dummy operation to print initial Knobs settings message
    // and force template to be loaded at start of loop
    theKnobs.handle_keypress('0');

    // initialize lookup table
    reload_template(theKnobs, theGHData, vfiles[nfile]);

    // and the image processing loop is running...
    bool is_running = true;

    while (is_running)
    {
        int kblur = theKnobs.get_pre_blur();
        int ksobel = theKnobs.get_ksize();

        // check for any operations that
        // might halt or reset the image processing loop
        if (theKnobs.get_op_flag(op_id))
        {
            if (op_id == Knobs::OP_TEMPLATE || op_id == Knobs::OP_UPDATE)
            {
                // changing the template will advance the file index
                if (op_id == Knobs::OP_TEMPLATE)
                {
                    nfile = (nfile + 1) % vfiles.size();
                }
                reload_template(theKnobs, theGHData, vfiles[nfile]);
            }
            else if (op_id == Knobs::OP_RECORD)
            {
                if (theKnobs.get_record_enabled())
                {
                    // reset recording frame counter
                    std::cout << "RECORDING STARTED" << std::endl;
                    n_record_ctr = 0;
                }
                else
                {
                    std::cout << "RECORDING STOPPED" << std::endl;
                }
            }
            else if (op_id == Knobs::OP_MAKE_VIDEO)
            {
                std::cout << "CREATING VIDEO FILE..." << std::endl;
                std::list<std::string> listOfPNG;
                get_dir_list(MOVIE_PATH, "*.png", listOfPNG);
                bool is_ok = make_video(15.0, MOVIE_PATH,
                    "movie.mov",
                    CV_FOURCC('M', 'P', '4', 'V'),
                    listOfPNG);
                std::cout << ((is_ok) ? "SUCCESS!" : "FAILURE!") << std::endl;
            }
        }

        // grab image
        vcap >> img;

        // apply the current image scale setting
        double img_scale = theKnobs.get_img_scale();
        Size viewer_size = Size(
            static_cast<int>(capture_size.width * img_scale),
            static_cast<int>(capture_size.height * img_scale));
        resize(img, img_viewer, viewer_size);
        
        // apply the current channel setting
        int nchan = theKnobs.get_channel();
        if (nchan == Knobs::ALL_CHANNELS)
        {
            // combine all channels into grayscale
            cvtColor(img_viewer, img_gray, COLOR_BGR2GRAY);
        }
        else
        {
            // select only one BGR channel
            split(img_viewer, img_channels);
            img_gray = img_channels[nchan];
        }
        
        // apply the current histogram equalization setting
        if (theKnobs.get_equ_hist_enabled())
        {
            double c = theKnobs.get_clip_limit();
            pCLAHE->setClipLimit(c);
            pCLAHE->apply(img_gray, img_gray);
        }

        // apply the current blur setting
        if (kblur > 1)
        {
            GaussianBlur(img_gray, img_gray, { kblur, kblur }, 0);
        }

        // create image of encoded Sobel gradient orientations from blurred input image
        // then apply Generalized Hough transform and locate maximum (best match)
        BGHMatcher::create_masked_gradient_orientation_img(img_gray, img_grad, theGHData.params);
        BGHMatcher::apply_ghough_transform_allpix<CV_16U, uint16_t>(img_grad, img_match, theGHData);

        minMaxLoc(img_match, nullptr, &qmax, nullptr, &ptmax);

        // apply the current output mode
        // content varies but all final output images are BGR
        switch (theKnobs.get_output_mode())
        {
            case Knobs::OUT_RAW:
            {
                // show the raw match result
                Mat temp_8U;
                normalize(img_match, img_match, 0, 255, cv::NORM_MINMAX);
                img_match.convertTo(temp_8U, CV_8U);
                cvtColor(temp_8U, img_viewer, COLOR_GRAY2BGR);
                break;
            }
            case Knobs::OUT_GRAD:
            {
                // display encoded gradient image
                // show red overlay of any matches that exceed arbitrary threshold
                Mat match_mask;
                std::vector<std::vector<cv::Point>> contours;
                normalize(img_grad, img_grad, 0, 255, cv::NORM_MINMAX);
                cvtColor(img_grad, img_viewer, COLOR_GRAY2BGR);
                normalize(img_match, img_match, 0, 1, cv::NORM_MINMAX);
                match_mask = (img_match > MATCH_DISPLAY_THRESHOLD);
                findContours(match_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
                drawContours(img_viewer, contours, -1, SCA_RED, -1, LINE_8, noArray(), INT_MAX);
                break;
            }
            case Knobs::OUT_PREP:
            {
                cvtColor(img_gray, img_viewer, COLOR_GRAY2BGR);
                break;
            }
            case Knobs::OUT_COLOR:
            default:
            {
                // no extra output processing
                break;
            }
        }

        // always show best match contour and target dot on BGR image
        image_output(img_viewer, qmax, ptmax, theKnobs, theGHData);

        // handle keyboard events and end when ESC is pressed
        is_running = wait_and_check_keys(theKnobs);
    }

    // when everything is done, release the capture device and windows
    vcap.release();
    destroyAllWindows();
}


int main(int argc, char** argv)
{
    loop();
    return 0;
}
