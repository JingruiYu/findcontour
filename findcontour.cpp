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
#define Filter 1

using namespace std;
using namespace cv;

Mat regoin, src, mask, src_gray;


int lowThreshold;
int max_lowThreshold = 100;
int angThreshold = 10;
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
                }
            }
        }
    }

    imshow("old canny",canny_output);

    vector< vector<Point> > vfront(37);
    vector< vector<Point> > vleft(37);
    vector< vector<Point> > vdown(37);
    vector< vector<Point> > vright(37);
    vector< vector<Point> > vvfront(37);
    vector< vector<Point> > vvleft(37);
    vector< vector<Point> > vvdown(37);
    vector< vector<Point> > vvright(37);
    int dt = 10;

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
                    int t = (floor(atan2(pt.y-py,pt.x-px) * 180 / pi) + 180 + 5) / dt;
                    int t2 = (floor(atan2(pt.y-py,pt.x-px) * 180 / pi) + 180) / dt;
                    vvleft[t2].push_back(pt);
                    vleft[t].push_back(pt); 
                }
                else if (regoin_y > 250) // up
                {
                    px = 191;
                    py = 132;
                    
                    Point pt(j,i);
                    int t = (floor(atan2(pt.y-py,pt.x-px) * 180 / pi) + 180 + 5) / dt;
                    int t2 = (floor(atan2(pt.y-py,pt.x-px) * 180 / pi) + 180) / dt;
                    vvfront[t2].push_back(pt);
                    vfront[t].push_back(pt);
                }
                else if (regoin_z > 250) // down
                {
                    px = 189;
                    py = 248;

                    Point pt(j,i);
                    int t = (floor(atan2(pt.y-py,pt.x-px) * 180 / pi) + 180 + 5) / dt;
                    int t2 = (floor(atan2(pt.y-py,pt.x-px) * 180 / pi) + 180) / dt;
                    vvdown[t2].push_back(pt);
                    vdown[t].push_back(pt);
                }
                else if (regoin_y > 98 && regoin_y < 103)
                {
                    px = 216;
                    py = 176;

                    Point pt(j,i);
                    int t = (floor(atan2(pt.y-py,pt.x-px) * 180 / pi) + 180 + 5) / dt;
                    int t2 = (floor(atan2(pt.y-py,pt.x-px) * 180 / pi) + 180) / dt;
                    vvright[t2].push_back(pt);
                    vright[t].push_back(pt);
                }
            }
        }
    }

    

#ifdef Filter
    int t1 = 100;
    int t2 = 100;

    for (size_t i = 0; i < vvright.size(); i++)
    {
        cv::Mat checkAngle = Mat::zeros( canny_output.size(), canny_output.type() );
        
        if (vvright[i].size() > t1)
        {
            cout << "vvright[" << i << "].size():" << vvright[i].size() << endl;
            // cout << "vvright[" << i << "].size():" << vright[i].size() << endl;
            for (size_t j = 0; j < vvright[i].size(); j++)
            {
                Point pt = vvright[i][j];

                /* int px = 183;
                int py = 118;
                                
                int t1 = floor(atan2(pt.y-py,pt.x-px) * 180 / pi)+180;
                
                cout << pt << " pt.y: " << pt.y << " t1: " << t1 << " i: " << i << endl; */
                checkAngle.at<uchar>(pt.y,pt.x) = 255;
            }
            imshow("angle",checkAngle);
            waitKey(0); 
        }
    }

    for (size_t i = 0; i < vvfront.size(); i++)
    {
        if (vvleft[i].size() > t1)
        {
            for (size_t j = 0; j < vvleft[i].size(); j++)
            {
                Point pt = vvleft[i][j];
                canny_output.at<uchar>(pt.y,pt.x) = 0;
            } 
        }

        if (vvfront[i].size() > t1)
        {
            for (size_t j = 0; j < vvfront[i].size(); j++)
            {
                Point pt = vvfront[i][j];
                canny_output.at<uchar>(pt.y,pt.x) = 0;
            }
        }

        if (vvdown[i].size() > t1) //20
        {
            for (size_t j = 0; j < vvdown[i].size(); j++)
            {
                Point pt = vvdown[i][j];
                canny_output.at<uchar>(pt.y,pt.x) = 0;
            }
        }

        if ( vvright[i].size() > t1 ) //30
        {
            for (size_t j = 0; j < vvright[i].size(); j++)
            {
                Point pt = vvright[i][j];
                canny_output.at<uchar>(pt.y,pt.x) = 0;
            }
        }
    } 

    for (size_t i = 0; i < vfront.size(); i++)
    {
        if (vleft[i].size() > t2)
        {
            for (size_t j = 0; j < vleft[i].size(); j++)
            {
                Point pt = vleft[i][j];
                canny_output.at<uchar>(pt.y,pt.x) = 0;
            } 
        }

        if (vfront[i].size() > t2)
        {
            for (size_t j = 0; j < vfront[i].size(); j++)
            {
                Point pt = vfront[i][j];
                canny_output.at<uchar>(pt.y,pt.x) = 0;
            }
        }

        if (vdown[i].size() > t2 ) // 10
        {
            for (size_t j = 0; j < vdown[i].size(); j++)
            {
                Point pt = vdown[i][j];
                canny_output.at<uchar>(pt.y,pt.x) = 0;
            }
        }

        if ( vright[i].size() > t2 ) //15
        {
            for (size_t j = 0; j < vright[i].size(); j++)
            {
                Point pt = vright[i][j];
                canny_output.at<uchar>(pt.y,pt.x) = 0;
            }
        }
    } 

    imshow("new canny",canny_output);
#endif
    

#ifdef DEBUG  
    imshow("new canny",canny_output);
    // waitKey(0);

    for (size_t i = 0; i < vvfront.size(); i++)
    {
        cv::Mat checkAngle = Mat::zeros( canny_output.size(), canny_output.type() );
        
        if (vvfront[i].size() != 0)
        {
            cout << "vvfront[" << i << "].size():" << vvfront[i].size() << endl;
            for (size_t j = 0; j < vvfront[i].size(); j++)
            {
                Point pt = vvfront[i][j];

                /* int px = 183;
                int py = 118;
                                
                int t1 = floor(atan2(pt.y-py,pt.x-px) * 180 / pi)+180;
                
                cout << pt << " pt.y: " << pt.y << " t1: " << t1 << " i: " << i << endl; */
                checkAngle.at<uchar>(pt.y,pt.x) = 255;
            }
            imshow("angle",checkAngle);
            waitKey(0); 
        }
    }
#endif
    
#ifdef Filter  
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

        if (contours[i].size() < 10)
            continue;
        
        // cannyContour = Mat::zeros( canny_output.size(), CV_8UC3 );
        Scalar color = Scalar( 255, 255, 255 );
        drawContours( cannyContour, contours, i, color, 1, 8, hierarchy, 0, Point() );

        cout << "contour size is : " << contours[i].size() << endl;
        
    }
    
    imshow( "cannyContour", cannyContour );
    waitKey(0);
#endif
}

void together(cv::Mat &canny, cv::Mat &free)
{
    // imshow("src",src);
    // imshow("canny",canny);
    // imshow("free",free);
    
    for (size_t i = 0; i < src.cols; i++)
    {
        for (size_t j = 0; j < src.rows; j++)
        {
            cv::Vec3b tmp  = canny.at<cv::Vec3b>(i,j);
            uchar regoin_x = tmp[0];
            
            if (regoin_x > 250)
                src.at<cv::Vec3b>(i,j) = cv::Vec3b(255,0,0);
            
            tmp = free.at<cv::Vec3b>(i,j);
            uchar regoin_y = tmp[0];
            if (regoin_y > 250)
                src.at<cv::Vec3b>(i,j) = cv::Vec3b(0,255,0);
        }
    }
    // imshow("srcNew",src);
    // waitKey(0);
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
        cout << "i: " << string(argv[1])+"/"+vstrBirdviewFilenames[i] << endl;
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
            cv::Mat cannyContour;
            int cannyTreshold = 50;
            GetContour(cannyContour,cannyTreshold);    

            imwrite( string(argv[1])+"/"+vstrBirdviewContourFilenames[i],cannyContour);
        }
               
        // waitKey(0);
	}


    return 0;
}
