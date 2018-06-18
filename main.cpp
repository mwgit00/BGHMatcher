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
#define DATA_PATH               "..\\TOGMatcher\\data\\"     // user may need to change this


using namespace cv;


#define SCA_BLACK   (cv::Scalar(0,0,0))
#define SCA_RED     (cv::Scalar(0,0,255))
#define SCA_GREEN   (cv::Scalar(0,255,0))
#define SCA_YELLOW  (cv::Scalar(0,255,255))


const char * stitle = "BGHMatcher";
int n_record_ctr = 0;


const std::vector<T_file_info> vfiles =
{
    { 0.05, "circle_b_on_w.png"},
    { 0.20, "bottle_20perc_top_b_on_w.png"},
    { 0.10, "outlet_cover.png"},
    { 0.10, "outlet_holes.png" },
    { 0.20, "panda_face.png"},
    { 0.05, "stars_main.png"}
};

size_t nfile = 0U;


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
    const Size& rsz = rtable.sz;
    const Point corner = { rptmax.x - rsz.width / 2, rptmax.y - rsz.height / 2 };

    // format score string for viewer (#.##)
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << (qmax / rtable.total_votes);

    // draw black background box
    // then draw text score on top of it
    rectangle(rimg, { 0,0,40,16 }, SCA_BLACK, -1);
    putText(rimg, oss.str(), { 0,12 }, FONT_HERSHEY_PLAIN, 1.0, SCA_GREEN, 1);

    // draw rectangle around best match
    rectangle(rimg, { corner.x, corner.y, rsz.width, rsz.height }, SCA_GREEN, 2);
    circle(rimg, rptmax, 2, SCA_YELLOW, -1);

    // save each frame to a file if recording
    if (rknobs.get_record_enabled())
    {
        std::ostringstream osx;
        osx << MOVIE_PATH << "img_" << std::setfill('0') << std::setw(5) << n_record_ctr << ".png";
        imwrite(osx.str(), rimg);
        n_record_ctr++;
        
        // red border around score box if recording
        rectangle(rimg, { 0,0,40,16 }, SCA_RED, 1);
    }
    
    cv::imshow(stitle, rimg);
}


void reload_template(BGHMatcher::T_ghough_table& rtable, const T_file_info& rinfo, const int ksize)
{
    const char * sxymtitle = "DX, DY, and Mask";
    const int KPAD = 4;
    const int KW = 480;
    const int KH = 160;
    Mat tdxdym = Mat::zeros({ KW, KH }, CV_8S);
    std::string spath = DATA_PATH + rinfo.sname;

    // clear the window
    imshow(sxymtitle, tdxdym);

    Mat img = imread(spath, IMREAD_GRAYSCALE);
    BGHMatcher::init_ghough_table_from_img(img, rtable, { ksize, BGHMatcher::BLUR_GAUSS, 1.0, rinfo.mag_thr });
    std::cout << "Loaded template (size= " << ksize << "): " << rinfo.sname << " " << rtable.total_votes << std::endl;

    // TODO -- display the template images
    
    cv::imshow(sxymtitle, tdxdym);
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
    Mat img_blur;
    Mat img_bgrad;
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
    reload_template(theGHData, vfiles[nfile], theKnobs.get_ksize());

    // and the image processing loop is running...
    bool is_running = true;

    while (is_running)
    {
        // check for any operations that
        // might halt or reset the image processing loop
        if (theKnobs.get_op_flag(op_id))
        {
            if (op_id == Knobs::OP_TEMPLATE || op_id == Knobs::OP_KSIZE)
            {
                // changing the template or template kernel size requires a reload
                // changing the template will advance the file index
                if (op_id == Knobs::OP_TEMPLATE)
                {
                    nfile = (nfile + 1) % vfiles.size();
                }
                reload_template(theGHData, vfiles[nfile], theKnobs.get_ksize());
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
        int kblur = theKnobs.get_pre_blur();
        BGHMatcher::blur_img(img_gray, img_blur, kblur, BGHMatcher::BLUR_GAUSS);

        // create the "binary gradient" image for blurred image
        // and apply mask for pixels at strong gradients
        BGHMatcher::cmp8NeighborsGT<uint8_t>(img_blur, img_bgrad);
        if (theGHData.params.mag_thr < 1.0)
        {
            BGHMatcher::apply_sobel_gradient_mask(img_gray, img_bgrad, kblur, theGHData.params.mag_thr / 2);
        }

        // apply Generalized Hough transform and locate maximum (best match)
        BGHMatcher::apply_ghough_transform<CV_16U, uint16_t>(img_bgrad, img_match, theGHData);
        minMaxLoc(img_match, nullptr, &qmax, nullptr, &ptmax);

        // apply the current output mode
        // content varies but all final output images are BGR
        switch (theKnobs.get_output_mode())
        {
            case Knobs::OUT_RAW:
            {
                // show the raw template match result
                Mat temp_8U;
                normalize(img_match, img_match, 0, 255, cv::NORM_MINMAX);
                img_match.convertTo(temp_8U, CV_8U);
                cvtColor(temp_8U, img_viewer, COLOR_GRAY2BGR);
                break;
            }
            case Knobs::OUT_MASK:
            {
                // display pre-processed gray input image
                // show red overlay of any matches that exceed arbitrary threshold
                Mat match_mask;
                std::vector<std::vector<cv::Point>> contours;
                cvtColor(img_bgrad, img_viewer, COLOR_GRAY2BGR);
                normalize(img_match, img_match, 0, 1, cv::NORM_MINMAX);
                match_mask = (img_match > MATCH_DISPLAY_THRESHOLD);
                findContours(match_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
                drawContours(img_viewer, contours, -1, SCA_RED, -1, LINE_8, noArray(), INT_MAX);
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