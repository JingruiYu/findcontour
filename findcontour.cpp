/*************************************************************************
	> File Name: findcontour.cpp
	> Author: 
	> Mail: 
	> Created Time: Wed 29 Jul 2020 09:45:31 PM EDT
 ************************************************************************/

#include<iostream>
#include <opencv2/opencv.hpp>

#define pi 3.1415926
// #define DEBUG
using namespace std;
using namespace cv;

Mat regoin, src, mask, src_gray;


int lowThreshold;
int max_lowThreshold = 100;
int angThreshold = 10;
char* window_name = "Edge Map";
bool findThreshold = false;

void LoadDataset(const string &strFile, vector<string> &vstrImageSaveFilenames, vector<string> &vstrBirdviewFilenames, 
                vector<string> &vstrBirdviewMaskFilenames, vector<string> &vstrBirdviewALLFilenames,
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
            vstrImageSaveFilenames.push_back("save/"+image);
            vstrBirdviewFilenames.push_back("birdview/"+image);
            vstrBirdviewMaskFilenames.push_back("mask/"+image);
            vstrBirdviewALLFilenames.push_back("all/"+image);
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

    int dt = 10;
    vector< vector < vector<Point> > > vPoints, vvPoints;
    vector< vector<Point> > vsubPoint(37);
    
    for (size_t i = 0; i < 4; i++)
    {
        vPoints.push_back(vsubPoint);
        vvPoints.push_back(vsubPoint);
    }
        

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
                    vPoints[0][t2].push_back(pt);
                    vvPoints[0][t].push_back(pt); 
                }
                else if (regoin_y > 250) // up
                {
                    px = 191;
                    py = 132;
                    
                    Point pt(j,i);
                    int t = (floor(atan2(pt.y-py,pt.x-px) * 180 / pi) + 180 + 5) / dt;
                    int t2 = (floor(atan2(pt.y-py,pt.x-px) * 180 / pi) + 180) / dt;
                    vPoints[1][t2].push_back(pt);
                    vvPoints[1][t].push_back(pt);
                }
                else if (regoin_z > 250) // down
                {
                    px = 189;
                    py = 248;

                    Point pt(j,i);
                    int t = (floor(atan2(pt.y-py,pt.x-px) * 180 / pi) + 180 + 5) / dt;
                    int t2 = (floor(atan2(pt.y-py,pt.x-px) * 180 / pi) + 180) / dt;
                    vPoints[2][t2].push_back(pt);
                    vvPoints[2][t].push_back(pt);
                }
                else if (regoin_y > 98 && regoin_y < 103) //right
                {
                    px = 216;
                    py = 176;

                    Point pt(j,i);
                    int t = (floor(atan2(pt.y-py,pt.x-px) * 180 / pi) + 180 + 5) / dt;
                    int t2 = (floor(atan2(pt.y-py,pt.x-px) * 180 / pi) + 180) / dt;
                    vPoints[3][t2].push_back(pt);
                    vvPoints[3][t].push_back(pt);
                }
            }
        }
    }

    /// threshold of size
    int t1 = 100;
    int t2 = 100;
#ifdef DEBUG
    for (size_t i = 0; i < 4; i++)
    {
        for (size_t k = 0; k < vPoints[i].size(); k++)
        {
            cv::Mat checkAngle = Mat::zeros( canny_output.size(), canny_output.type() );
        
            if (vPoints[i][k].size() > t1)
            {
                cout << "vPoints[" << i << "][" << k << "].size(): " << vPoints[i][k].size() << endl;
                for (size_t j = 0; j < vPoints[i][k].size(); j++)
                {
                    Point pt = vPoints[i][k][j];

                    checkAngle.at<uchar>(pt.y,pt.x) = 255;
                }
                imshow("dt angle",checkAngle);
                waitKey(0); 
            }
        }        
    }
#endif

    for (size_t k = 0; k < 4; k++)
    {
        for (size_t i = 0; i < vPoints[k].size(); i++)
        {
            if (vPoints[k][i].size() > t1)
            {
                for (size_t j = 0; j < vPoints[k][i].size(); j++)
                {
                    Point pt = vPoints[k][i][j];
                    canny_output.at<uchar>(pt.y,pt.x) = 0;
                }                
            }
        }

        for (size_t i = 0; i < vvPoints[k].size(); i++)
        {
            if (vvPoints[k][i].size() > t1)
            {
                for (size_t j = 0; j < vvPoints[k][i].size(); j++)
                {
                    Point pt = vvPoints[k][i][j];
                    canny_output.at<uchar>(pt.y,pt.x) = 0;
                }                
            }
        }
    }
    

    findContours( canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0) );
    
    cannyContour = Mat::zeros( canny_output.size(), CV_8UC3 );
    for( int i = 0; i< contours.size(); i++ )
	{
        if (contours[i].size() < 10)
            continue;

#ifdef DEBUG      
        cannyContour = Mat::zeros( canny_output.size(), CV_8UC3 );
#endif
        Scalar color = Scalar( 255, 255, 255 );
        drawContours( cannyContour, contours, i, color, 1, 8, hierarchy, 0, Point() );        
    }
    
#ifdef DEBUG
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
    vector<string> vstrBirdviewFilenames, vstrBirdviewMaskFilenames, vstrImageSaveFilenames, vstrBirdviewALLFilenames;
	vector<double> vTimestamps;
	vector<cv::Vec3d> vodomPose;

	string DataStrFile = string(argv[1])+"/associate.txt";
	
    LoadDataset(DataStrFile, vstrImageSaveFilenames, vstrBirdviewFilenames, vstrBirdviewMaskFilenames, vstrBirdviewALLFilenames, vodomPose, vTimestamps);

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
            waitKey(0);
        }
        else
        {
            cv::Mat cannyContour;
            int cannyTreshold = 50;
            GetContour(cannyContour,cannyTreshold);    

            imwrite( string(argv[1])+"/"+vstrImageSaveFilenames[i],cannyContour);
        }
               
        // waitKey(0);
	}


    return 0;
}
