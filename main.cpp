#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/types.hpp>
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/videoio.hpp>


const double kDistanceCoef = 4.0;
const int kMaxMatchingSize = 50;

std::vector<cv::Scalar> DefColors() {
    std::vector<cv::Scalar> colors;
    colors.push_back(cv::Scalar( 0,0,255));
    colors.push_back(cv::Scalar( 255,0,255));
    colors.push_back(cv::Scalar( 0,255,255));
    colors.push_back(cv::Scalar( 255,255,0));

    return colors;
}

inline void detect_and_compute(cv::Mat& img, std::vector<cv::KeyPoint>& kpts, cv::Mat& desc) {
    cv::Ptr<cv::Feature2D> sift = cv::SIFT::create();
    sift->detectAndCompute(img, cv::Mat(), kpts, desc);
}

inline void match(cv::Mat& desc1, cv::Mat& desc2, std::vector<cv::DMatch>& matches) {
    matches.clear();
    cv::BFMatcher desc_matcher(cv::NORM_L2, true);
    std::vector< std::vector<cv::DMatch> > vmatches;
    desc_matcher.knnMatch(desc1, desc2, vmatches, 1);
    for (int i = 0; i < static_cast<int>(vmatches.size()); ++i) {
        if (!vmatches[i].size()) {
            continue;
        }
        matches.push_back(vmatches[i][0]);

    }
    std::sort(matches.begin(), matches.end());
    while (matches.front().distance * kDistanceCoef < matches.back().distance) {
        matches.pop_back();
    }
    while (matches.size() > kMaxMatchingSize) {
        matches.pop_back();
    }
}

inline cv::Mat findKeyPointsHomography(std::vector<cv::KeyPoint>& kpts1, std::vector<cv::KeyPoint>& kpts2,
                                       std::vector<cv::DMatch>& matches, std::vector<char>& match_mask) {
    std::vector<cv::Point2f> pts1;
    std::vector<cv::Point2f> pts2;
    for (int i = 0; i < static_cast<int>(matches.size()); ++i) {
        pts1.push_back(kpts1[matches[i].queryIdx].pt);
        pts2.push_back(kpts2[matches[i].trainIdx].pt);
    }
    cv::Mat H = findHomography(pts1, pts2, cv::RANSAC, 4, match_mask);
    return H;
}



int main() {
    cv::VideoCapture cap("../video.mov");
    std::vector<cv::Mat> images;
    cv::Mat image;
    std::vector<cv::String> filenames;
    std::vector<cv::Scalar> colors;
    cv::utils::fs::glob(cv::String("../objects"),cv::String("*.png"),filenames);
    for (const auto& fn: filenames) {
        image = cv::imread(fn);
        images.push_back(image);
    }
    colors = DefColors();
    if(cap.isOpened()) // check if we succeeded
    {
        cv::Mat frame,ret, old_frame;
        cap.read(old_frame);
        ret = old_frame.clone();
        for (int j = 0; j < images.size(); j++)
        {
            std::vector<cv::KeyPoint> kpts1;
            std::vector<cv::KeyPoint> kpts2;

            cv::Mat desc1;
            cv::Mat desc2;

            std::vector<cv::DMatch> matches;

            detect_and_compute(images[j], kpts1, desc1);
            detect_and_compute(old_frame, kpts2, desc2);

            match( desc1, desc2, matches);

            std::vector<char> match_mask(matches.size(), 1);
            cv::Mat H = findKeyPointsHomography(kpts1, kpts2, matches, match_mask);

            cv::Mat res;
            cv::drawMatches(images[j], kpts1, old_frame, kpts2, matches, res, cv::Scalar::all(-1),
                            cv::Scalar::all(-1), match_mask, cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            std::vector<cv::Point2f> obj_corners(4);
            obj_corners[0] = cv::Point2f(0, 0);
            obj_corners[1] = cv::Point2f((float) images[j].cols, 0);
            obj_corners[2] = cv::Point2f((float) images[j].cols, (float) images[j].rows);
            obj_corners[3] = cv::Point2f(0, (float) images[j].rows);
            std::vector<cv::Point2f> scene_corners(4);
            perspectiveTransform(obj_corners, scene_corners, H);
            //-- Draw lines between the corners (the mapped object in the scene - image_2 )
            line(res, scene_corners[0] + cv::Point2f((float) images[j].cols, 0),
                 scene_corners[1] + cv::Point2f((float) images[j].cols, 0), colors[j], 4);
            line(res, scene_corners[1] + cv::Point2f((float) images[j].cols, 0),
                 scene_corners[2] + cv::Point2f((float) images[j].cols, 0), colors[j], 4);
            line(res, scene_corners[2] + cv::Point2f((float) images[j].cols, 0),
                 scene_corners[3] + cv::Point2f((float) images[j].cols, 0), colors[j], 4);
            line(res, scene_corners[3] + cv::Point2f((float) images[j].cols, 0),
                 scene_corners[0] + cv::Point2f((float) images[j].cols, 0), colors[j], 4);
            cv::imshow("result", res);
            cv::waitKey(0);
        }
        std::vector<cv::Point2f> features_prev, features_next;
        std::vector<uchar> status;
        std::vector<float> err;
    }
    return 0;
}


