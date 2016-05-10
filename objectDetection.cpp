#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

bool isPixelInside(int row,int col,int x,int y);
Mat diffsizekernel(Mat img, int f, int c);
Mat diffx(Mat img);
Mat diffy(Mat img); 
string type2str(int type);

bool isPixelInside(int row,int col,int x,int y)
{
	if(x < row && y < col && x >= 0 && y >= 0)
		return true;
	return false;
}


Mat diffsizekernel(Mat img, int f, int c) 
{
    float dkernel[] =  {-1, 0, 1};

    Mat kernel = Mat(f, c, CV_32FC1, dkernel);

    Mat imgDiff;
	
    filter2D( img, imgDiff,-1,kernel,Point(-1,-1) );

    return imgDiff;
}

Mat diffx(Mat img) 
{
    return diffsizekernel(img, 3, 1);
}

Mat diffy(Mat img) 
{
    return diffsizekernel(img, 1, 3);
}


string type2str(int type)
{
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}




int main(int argc,char **argv)
{
	float mask[] = {-1,0,1};
	Mat image = imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
    image.convertTo(image,CV_64F);
    Mat x = diffx(image);
    Mat y = diffy(image);
	Mat magnitude(x.size(),x.type());
	Mat angle(x.size(),x.type());
	Mat absx,absy;
	std::vector<int> params;
	params.push_back(CV_IMWRITE_JPEG_QUALITY);
	params.push_back(100);
	
	
	
	string s = type2str(x.type());
	cout<<"\n\n\ntype : "<<s<<"\n\n\n";
	
	
	
	if((x.size() == y.size()) && (x.type() == y.type()) && (x.depth() == CV_32F || x.depth() == CV_64F)) 
		cout<<"\n\n\nall ok\t"<<"type : "<<x.type()<<"\n\n\n";
	else
		cout<<"\n\n\nall not ok\t"<<"type : "<<x.depth()<<"\n\n\n";
	cartToPolar(x,y,magnitude,angle,true);
	
	
	convertScaleAbs( x, absx );
	convertScaleAbs( x, absy );
	imwrite("changeX.jpg",x,params);
	imwrite("changeY.jpg",y,params);
	imwrite("changeAbsX.jpg",absx,params);
	imwrite("changeAbsY.jpg",absy,params);
	imwrite("magnitude.jpg",magnitude,params);
	imwrite("angle.jpg",angle,params);
	    
    return 0;
}


