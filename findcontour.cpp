/*************************************************************************
	> File Name: findcontour.cpp
	> Author: 
	> Mail: 
	> Created Time: Wed 29 Jul 2020 09:45:31 PM EDT
 ************************************************************************/

#include<iostream>
#include <opencv2/opencv.hpp>

#define pi 3.1415926
// #define DEBUG 1

using namespace std;
using namespace cv;

Mat regoin, src, mask, src_gray;


int lowThreshold;
int max_lowThreshold = 100;
char* window_name = "Edge Map";
bool findThreshold = false;

void LoadDataset(const string &strFile, vector<string> &vstrImageFilenames, vector<string> &vstrBirdviewFilenames, 
                vector<string> &vstrBirdviewMaskFilenames, vector<string> &vstrBirdviewContourFilenames,
                vector<cv::Vec3d> &vodomPose, vector<double> &vTimestamps)
{
    ifstream f;
    f.open(strFile.c_str());

    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            double x,y,theta;
            string image;
            ss >> t;
            vTimestamps.push_back(t);
            ss>>x>>y>>theta;
            vodomPose.push_back(cv::Vec3d(x,y,theta));
            ss >> image;
            vstrImageFilenames.push_back("image/"+image);
            vstrBirdviewFilenames.push_back("birdview/"+image);
            vstrBirdviewMaskFilenames.push_back("mask/"+image);
            vstrBirdviewContourFilenames.push_back("contour/"+image);
        }
    }
}


void CannyThreshold(int, void*)
{
    Mat canny_output;
    vector< vector<Point> > contours;
    vector<Vec4i> hierarchy;

    cv::Mat detected_edges;

    int ratio = 3;
    int kernel_size = 3;

    /// 使用 3x3内核降噪
    blur( src_gray, detected_edges, Size(3,3) );

    /// 运行Canny算子
    Canny( detected_edges, canny_output, lowThreshold, lowThreshold*ratio, kernel_size );

    findContours( canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    cv::Mat dst;
    /// 创建与src同类型和大小的矩阵(dst)
    dst.create( src.size(), src.type() );
    /// 使用 Canny算子输出边缘作为掩码显示原图像
    dst = Scalar::all(0);
    src.copyTo( dst, canny_output);

    RNG rng(12345);

    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
    for( int i = 0; i< contours.size(); i++ )
	{
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        drawContours( drawing, contours, i, color, CV_FILLED, 8, hierarchy, 0, Point() );
    }
    
    imshow( window_name, drawing );
}


void filterContour(vector< vector<Point> > &contours, vector< vector<Point> > &contoursFilter)
{
    for ( int i = 0; i < contours.size(); i++)
    {
        if (contours[i].size() > 10)
        {
            float maskednum = 0;
            float blacknum = 0;
            vector<float> vtheta;
            float thsum = 0;
            for (size_t j = 0; j < contours[i].size(); j++)
            {
                Point pt = contours[i][j]; //  pt.x

                cv::Vec3b tmp = mask.at<cv::Vec3b>(i,j);
                uchar mask_y = tmp[1];
                int mask_y_int = (int) mask_y;
                // if y == 255, it is avaliable;
                if (mask_y_int != 255)
                    maskednum = maskednum + 1.0;

                tmp = regoin.at<cv::Vec3b>(i,j);
                uchar regoin_x = tmp[0];
                uchar regoin_y = tmp[1];
                uchar regoin_z = tmp[2];
                // int regoin_y_int = (int) regoin_y;

                if ( regoin_x < 10 && regoin_y < 10 && regoin_z < 10 )
                    blacknum = blacknum + 1.0;

                int px,py;
                if (regoin_x > 250) // left
                {
                    px = 163;
                    py = 175;
                }
                else if (regoin_y > 250) // up
                {
                    px = 189;
                    py = 129;
                }
                else if (regoin_z > 250) // down
                {
                    px = 189;
                    py = 248;
                }
                else // right
                {
                    px = 216;
                    py = 176;
                }

                float theta;
                theta = (py - pt.y) / (px - pt.x);
                thsum += theta;
                vtheta.push_back(theta);
            }
            
            if (maskednum > 10 || maskednum / contours[i].size() > 0.3)
                continue;
            
            if (blacknum / contours[i].size() > 0.7)
                continue;
            
            float thmean = thsum / vtheta.size();

            float thstd = 0;
            for (size_t k = 0; k < vtheta.size(); k++)
                thstd += (vtheta[k] - thmean)*(vtheta[k] - thmean);
            
            thstd = sqrt(thstd/(vtheta.size()-1));
            
            if (thstd < 0.1) // 
                continue;
            
            contoursFilter.push_back(contours[i]);
        }
    }
    
}


void GetContour(cv::Mat &cannyContour, int cannyTreshold)
{
    Mat canny_output;
    vector< vector<Point> > contours;
    vector< vector<Point> > contoursFilter;
    vector<Vec4i> hierarchy;

    cv::Mat detected_edges;

    int ratio = 3;
    int kernel_size = 3;

    /// 使用 3x3内核降噪
    blur( src_gray, detected_edges, Size(3,3) );

    /// 运行Canny算子
    Canny( detected_edges, canny_output, cannyTreshold, cannyTreshold*ratio, kernel_size );


#ifdef DEBUG    
    imshow("old canny",canny_output);
#endif

    vector< vector<Point> > vfront(361);
    vector< vector<Point> > vleft(361);
    vector< vector<Point> > vdown(361);
    vector< vector<Point> > vright(361);

    for (size_t i = 0; i < canny_output.cols; i++)
    {
        for (size_t j = 0; j < canny_output.rows; j++)
        {
            if (canny_output.at<uchar>(i,j) > 250)
            {
                cv::Vec3b tmp  = regoin.at<cv::Vec3b>(i,j);
                uchar regoin_x = tmp[0];
                uchar regoin_y = tmp[1];
                uchar regoin_z = tmp[2];

                if (mask.at<cv::Vec3b>(i,j)[1] < 250 || 
                    (regoin_x + regoin_y + regoin_z) < 10 )
                {
                    canny_output.at<uchar>(i,j) = 0;
                    continue;
                }

                int px,py;
                if (regoin_x > 250) // left
                {
                    px = 163;
                    py = 175;

                    Point pt(j,i);
                    int t = floor(atan2(pt.y-py,pt.x-px) * 180 / pi)+180;
                    vleft[t].push_back(pt);  
                }
                else if (regoin_y > 250) // up
                {
                    px = 183;
                    py = 118;
                    
                    Point pt(j,i);
                    int t = floor(atan2(pt.y-py,pt.x-px) * 180 / pi)+180;
                    vfront[t].push_back(pt);
                }
                else if (regoin_z > 250) // down
                {
                    px = 189;
                    py = 248;

                    Point pt(j,i);
                    int t = floor(atan2(pt.y-py,pt.x-px) * 180 / pi)+180;
                    vdown[t].push_back(pt);
                }
                else if (regoin_y > 98 && regoin_y < 103)
                {
                    px = 216;
                    py = 176;

                    Point pt(j,i);
                    int t = floor(atan2(pt.y-py,pt.x-px) * 180 / pi)+180;
                    vright[t].push_back(pt);
                }
            }
        }
    }

    for (size_t i = 0; i < vfront.size(); i++)
    {
        if (vleft[i].size() > 10)
        {
            for (size_t j = 0; j < vleft[i].size(); j++)
            {
                Point pt = vleft[i][j];
                canny_output.at<uchar>(pt.y,pt.x) = 0;
            }
        }

        if (vfront[i].size() > 10)
        {
            for (size_t j = 0; j < vfront[i].size(); j++)
            {
                Point pt = vfront[i][j];
                canny_output.at<uchar>(pt.y,pt.x) = 0;
            }
        }

        if (vdown[i].size() > 10)
        {
            for (size_t j = 0; j < vdown[i].size(); j++)
            {
                Point pt = vdown[i][j];
                canny_output.at<uchar>(pt.y,pt.x) = 0;
            }
        }

        if (vright[i].size() > 10)
        {
            for (size_t j = 0; j < vright[i].size(); j++)
            {
                Point pt = vright[i][j];
                canny_output.at<uchar>(pt.y,pt.x) = 0;
            }
        }
    }
    
    

#ifdef DEBUG    
    imshow("new canny",canny_output);
    waitKey(0);

    for (size_t i = 0; i < vright.size(); i++)
    {
        cv::Mat checkAngle = Mat::zeros( canny_output.size(), canny_output.type() );
        
        if (vright[i].size() != 0)
        {
            cout << "vright[" << i << "].size():" << vright[i].size() << endl;
            for (size_t j = 0; j < vright[i].size(); j++)
            {
                Point pt = vright[i][j];

                int px = 183;
                int py = 118;
                                
                int t1 = floor(atan2(pt.y-py,pt.x-px) * 180 / pi)+180;
                
                cout << pt << " pt.y: " << pt.y << " t1: " << t1 << " i: " << i << endl;
                checkAngle.at<uchar>(pt.y,pt.x) = 255;
            }
            imshow("angle",checkAngle);
            waitKey(0); 
        }
    }
#endif
    

    findContours( canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    RNG rng(12345);

    // filterContour(contours, contoursFilter);
    
    cannyContour = Mat::zeros( canny_output.size(), CV_8UC3 );
    for( int i = 0; i< contours.size(); i++ )
	{
        // Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        // drawContours( cannyContour, contours, i, color, CV_FILLED, 8, hierarchy, 0, Point() );

        vector<float> vtheta;
        float thsum = 0;

        if (contours[i].size() < 15)
            continue;
        
        cannyContour = Mat::zeros( canny_output.size(), CV_8UC3 );
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        drawContours( cannyContour, contours, i, color, CV_FILLED, 8, hierarchy, 0, Point() );

        imshow( "cannyContour", cannyContour );
        waitKey(0);
    }
    
}


int main(int argc, char const *argv[])
{
    vector<string> vstrBirdviewFilenames, vstrBirdviewMaskFilenames, vstrImageFilenames, vstrBirdviewContourFilenames;
	vector<double> vTimestamps;
	vector<cv::Vec3d> vodomPose;

	string DataStrFile = string(argv[1])+"/associate.txt";
	
    LoadDataset(DataStrFile, vstrImageFilenames, vstrBirdviewFilenames, vstrBirdviewMaskFilenames, vstrBirdviewContourFilenames, vodomPose, vTimestamps);

    regoin = imread(string(argv[1])+"/regoin.jpg",CV_LOAD_IMAGE_UNCHANGED);

    for (size_t i = 0; i < vstrBirdviewFilenames.size(); i++)
	{	
        cout << "i: " << i << endl;
		/// 装载图像
		src = imread(string(argv[1])+"/"+vstrBirdviewFilenames[i],CV_LOAD_IMAGE_UNCHANGED);
		mask = imread(string(argv[1])+"/"+vstrBirdviewMaskFilenames[i],CV_LOAD_IMAGE_UNCHANGED);
		
        cvtColor( src, src_gray, CV_BGR2GRAY );

        if (findThreshold)
        {
            namedWindow( window_name, CV_WINDOW_AUTOSIZE );
            /// 创建trackbar
            createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold );
            /// 显示图像
            CannyThreshold(0, 0); 
        }
        else
        {
            // imshow("src", src);
            cv::Mat cannyContour;
            int cannyTreshold = 50;
            GetContour(cannyContour,cannyTreshold);    

            // imshow( "Mask add", regoin );
            // imwrite( string(argv[1])+"/"+vstrBirdviewContourFilenames[i],cannyContour);
        }
               
        waitKey(0);
	}


    return 0;
}
