#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/core/types.hpp>
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
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
cv::Mat Draw_rectangle(cv::Mat res,std::vector<cv::Point2f>obj_corners, cv::Scalar color)
{

    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line(res, obj_corners[0],obj_corners[1], color, 4);
    line(res, obj_corners[1],obj_corners[2], color, 4);
    line(res, obj_corners[2],obj_corners[3], color, 4);
    line(res, obj_corners[3],obj_corners[0], color, 4);
    return res;

}


std::vector<cv::Point2f> FindCorners(cv::Mat &image, cv::Mat H, cv::Mat res, cv::Scalar colors) {

    std::vector<cv::Point2f> obj_corners(4);
    obj_corners[0] = cv::Point2f(0, 0);
    obj_corners[1] = cv::Point2f((float) image.cols, 0);
    obj_corners[2] = cv::Point2f((float) image.cols, (float) image.rows);
    obj_corners[3] = cv::Point2f(0, (float) image.rows);

    std::vector<cv::Point2f> scene_corners(4);
    perspectiveTransform(obj_corners, scene_corners, H);
    obj_corners[0]=scene_corners[0] + cv::Point2f((float) image.cols, 0);
    obj_corners[1]=scene_corners[1] + cv::Point2f((float) image.cols, 0);
    obj_corners[2]=scene_corners[2] + cv::Point2f((float) image.cols, 0);
    obj_corners[3]=scene_corners[3] + cv::Point2f((float) image.cols, 0);
    res = Draw_rectangle(res,obj_corners,colors);
    cv::imshow("result", res);

    cv::waitKey(0);
    return scene_corners;
}


std::vector<cv::Point2f> FindPointsForTrack(std::vector<cv::Point2f> scene_corners) {
    cv::Point2f center, second_point;
    std::vector<cv::Point2f> points(2);
    center = cv::Point2f(scene_corners[0].x+(scene_corners[2].x-scene_corners[0].x)/2,scene_corners[0].y-(scene_corners[0].y-scene_corners[2].y)/2);
    points[0] = center;
    second_point = cv::Point2f(scene_corners[0].x+(center.x-scene_corners[0].x)/2,scene_corners[0].y-(scene_corners[0].y-center.y)/2);
    points[1] = second_point;
    return points;

}

cv::Mat Draw_rectangle_by_center(std::vector<cv::Point2f> corners, cv::Mat image, std::vector<cv::Scalar> colors,
                                 std::vector<float> height, std::vector<float> width)
{
    std::vector<cv::Point2f> obj_corners(4);
    cv::Point2f center,second_point;
    int objectC = 0;
    for(int i=0;i<4;i++)
    {
        center = corners[objectC];
        objectC++;
        second_point = corners[objectC];
        objectC++;
        float x,y;
        x = center.x -second_point.x;
        y = second_point.y - center.y;
        obj_corners[0] = cv::Point2f(center.x-2*x,center.y+2*y);
        obj_corners[2] = cv::Point2f(center.x+2*x,center.y-2*y);
        obj_corners[1] = cv::Point2f(obj_corners[2].x-width[i],obj_corners[0].y-height[i]);
        obj_corners[3] = cv::Point2f(obj_corners[0].x+width[i],obj_corners[2].y+height[i]);
        image = Draw_rectangle(image,obj_corners,colors[i]);
    }
    return image;
}


int main() {
    cv::VideoCapture cap("../video.mov");
    std::vector<cv::Mat> images;
    std::vector<cv::Point2f> corners(16), corners_next(16);
    cv::Mat image;
    std::vector<cv::String> filenames;
    cv::utils::fs::glob(cv::String("../objects"),cv::String("*.png"),filenames);
    std::vector<cv::Point2f> obj_corners(4), obj_corners_next(4);
    std::vector<cv::Scalar> colors;
    std::vector<float> height(4), width(4);
    colors = DefColors();
    for (const auto& fn: filenames) {
        image = cv::imread(fn);
        images.push_back(image);
    }
    if(cap.isOpened()) // check if we succeeded
    {
        cv::Mat ret, frame;
        cap.read(frame);
        int objectC = 0;
        for (int j = 0; j < images.size(); j++)
        {
            std::vector<cv::KeyPoint> kpts1;
            std::vector<cv::KeyPoint> kpts2;
            std::vector<cv::Point2f> scene_corners;

            cv::Mat desc1;
            cv::Mat desc2;
            std::vector<cv::DMatch> matches;

            detect_and_compute(images[j], kpts1, desc1);
            detect_and_compute(frame, kpts2, desc2);

            match( desc1, desc2, matches);

            std::vector<char> match_mask(matches.size(), 1);
            cv::Mat H = findKeyPointsHomography(kpts1, kpts2, matches, match_mask);

            cv::Mat res;
            cv::drawMatches(images[j], kpts1, frame, kpts2, matches, res, cv::Scalar::all(-1),
                            cv::Scalar::all(-1), match_mask, cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

            scene_corners = FindCorners(images[j],H, res, colors[j]);
            std::vector<cv::Point2f> points;
            height[j] = scene_corners[0].y-scene_corners[1].y;
            width[j] = scene_corners[2].x-scene_corners[1].x;
            points = FindPointsForTrack(scene_corners);

            corners[objectC] = points[0];
            objectC++;
            corners[objectC] = points[1];
            objectC++;
        }
        cv::destroyWindow("result");

        cv:: Mat gray,prevGray;
        cv::Size subPixWinSize(10,10), winSize(31,31);
        for(;;) {
            cv::TermCriteria termcrit(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,20,0.03);
            cap >> frame;
            if (frame.empty())
                break;

            frame.copyTo(image);
            cvtColor(image, gray, cv::COLOR_BGR2GRAY);
            std::vector<uchar> status;
            std::vector<float> err;
            if(prevGray.empty())
                gray.copyTo(prevGray);
            calcOpticalFlowPyrLK(prevGray, gray, corners, corners_next, status, err, winSize,
                                 20, termcrit, 0, 0.001);

            image = Draw_rectangle_by_center(corners,image,colors,height,width);
            for(int i=0; i<corners_next.size();i++)
            {
                cv::circle(image, corners_next[i] ,3, cv::Scalar(0,255,0), -1, 8);
            }
            cv::imshow("LK Demo", image);
            cv::waitKey(1);
            std::swap(corners, corners_next);
            cv::swap(prevGray, gray);
        }

    }
    return 0;
}
