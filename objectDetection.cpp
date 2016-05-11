#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

#include <iostream>
#include <stdio.h>
#include <cmath>

using namespace std;
using namespace cv;

bool isPixelInside(int row,int col,int x,int y);
Mat diffsizekernel(Mat img, int f, int c);
Mat diffx(Mat img);
Mat diffy(Mat img); 
string type2str(int type);
Mat remRowCol(Mat img,int r,int c);
Mat addPi(Mat angle);
std::vector< std::vector<double> > histogramOfCells(Mat angle,Mat gradient);
std::vector<double> binning(Mat angle,Mat gradient,int x,int y);
Mat gaussianSpatialWindow(Mat img);


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

Mat diffy(Mat img) 
{
	return diffsizekernel(img, 3, 1);
}

Mat diffx(Mat img) 
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
	default:	 r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

Mat remRowCol(Mat img,int r,int c)
{
	int col = img.cols;
	int row = img.rows;
	
	Mat image(row - r, col - c,CV_32FC1);
	
	int i,j;
	for(j=c/2;j < (col - c/2);j++)
		for(i=r/2;i<(row - r/2);i++)
			image.at<float>(i,j) = img.at<float>(i,j+1);
	  
	return image;
	
}


Mat addPi(Mat angle)
{

	int col = angle.cols;
	int row = angle.rows;
	
	int i,j;
	for(j=0;j < col;j++)
		for(i=0;i<row;i++)
			if(angle.at<float>(i,j) > 180.0)
				angle.at<float>(i,j) = angle.at<float>(i,j) - 180;
	return angle;

}

std::vector< std::vector<double> > histogramOfCells(Mat angle,Mat gradient)
{
	std::vector< std::vector<double> > cellHist;
	std::vector<double> hist;
	int col = angle.cols;
	int row = angle.rows;
	int i,j;
	for(i = 0; i <row;i += 6)
		for(j=0;j < col; j += 6)
		{
			hist = binning(angle,gradient,i,j);
			cellHist.push_back(hist);
		}
	
	return cellHist;
	
	
}

std::vector<double> binning(Mat angle,Mat gradient,int x,int y)
{
	std::vector<double> hist(9, 0);
	int j,i;
	for(i = x; i <x+6;i++)
		for(j=y;j < y+6; j++)
		{
			if(angle.at<float>(i,j) < 20.0)
				hist[0] += gradient.at<float>(i,j);
			else if(angle.at<float>(i,j) < 40.0)
				hist[1] += gradient.at<float>(i,j);
			else if(angle.at<float>(i,j) < 60.0)
				hist[2] += gradient.at<float>(i,j);
			else if(angle.at<float>(i,j) < 80.0)
				hist[3] += gradient.at<float>(i,j);
			else if(angle.at<float>(i,j) < 100.0)
				hist[4] += gradient.at<float>(i,j);
			else if(angle.at<float>(i,j) < 120.0)
				hist[5] += gradient.at<float>(i,j);
			else if(angle.at<float>(i,j) < 140.0)
				hist[6] += gradient.at<float>(i,j);
			else if(angle.at<float>(i,j) < 160.0)
				hist[7] += gradient.at<float>(i,j);
			else
				hist[8] += gradient.at<float>(i,j);
				
		}
	return hist;
}

Mat gaussianSpatialWindow(Mat img)
{
	Mat kernel = (Mat_<double>(6,6) <<  0.025291,0.026736,0.027489,0.027489,0.026736,0.025291,
										0.026736,0.028263,0.029059,0.029059,0.028263,0.026736,
										0.027489,0.029059,0.029878,0.029878,0.029059,0.027489,
										0.027489,0.029059,0.029878,0.029878,0.029059,0.027489,
										0.026736,0.028263,0.029059,0.029059,0.028263,0.026736,
										0.025291,0.026736,0.027489,0.027489,0.026736,0.025291);

	Mat imgDiff;
	filter2D( img, imgDiff,-1,kernel,Point(-1,-1) );
	return imgDiff;
}

int main(int argc,char **argv)
{
	const int cellSize = 6;
	double min,max;
	Mat image = imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
	
	image.convertTo(image,CV_32F);
	if(image.rows%cellSize != 0 || image.cols%cellSize == 0)
		image = remRowCol(image,image.rows%cellSize,image.cols%cellSize);
	
	Mat x = diffx(image);
	Mat y = diffy(image);
	Mat magnitude(x.size(),x.type());
	Mat angle(x.size(),x.type());
	Mat absx,absy;
	
	std::vector<int> params;
	
	params.push_back(CV_IMWRITE_JPEG_QUALITY);
	params.push_back(100);
	
	
	cartToPolar(x,y,magnitude,angle,true);
	
	angle = addPi(angle);
	magnitude = gaussianSpatialWindow(magnitude);
	
	minMaxLoc(angle,&min,&max);
	
	cout<<"\n\n\nmin = "<<min<<"\nmax = "<<max<<"\n\n\n";
	
	
	convertScaleAbs( x, absx );
	convertScaleAbs( y, absy );
	imwrite("changeX.jpg",x,params);
	imwrite("changeY.jpg",y,params);
	imwrite("changeAbsX.jpg",absx,params);
	imwrite("changeAbsY.jpg",absy,params);
	imwrite("magnitude.jpg",magnitude,params);
	imwrite("angle.jpg",angle,params);
/*
	cin.ignore();
	remove("changeX.jpg");
	remove("changeY.jpg");
	remove("changeAbsX.jpg");
	remove("changeAbsY.jpg");
	remove("magnitude.jpg");
	remove("angle.jpg");
*/	
	return 0;
}


