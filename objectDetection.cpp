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
std::vector< std::vector< std::vector<double> > > histogramOfCells(Mat angle,Mat gradient);
std::vector<double> binning(Mat angle,Mat gradient,int x,int y);
Mat gaussianSpatialWindow(Mat img,int blockSize);
std::vector< std::vector< std::vector<double> > > getBlockDiscriptors(std::vector< std::vector< std::vector<double> > > cellHistograms);
std::vector<double> getSingleBlockDiscriptor(std::vector< std::vector< std::vector<double> > > cellHistograms,int x,int y);

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

std::vector< std::vector< std::vector<double> > > histogramOfCells(Mat angle,Mat gradient)
{
	std::vector< std::vector< std::vector<double> > > cellHist;
	std::vector< std::vector<double> > cellHistRow;
	std::vector<double> hist;
	int col = angle.cols;
	int row = angle.rows;
	int i,j;
	for(i = 0; i <row;i += 6)
	{
		for(j=0;j < col; j += 6)
		{
			hist = binning(angle,gradient,i,j);
			cellHistRow.push_back(hist);
		}
		cellHist.push_back(cellHistRow);
	}
	return cellHist;
	
	
}

std::vector<double> binning(Mat angle,Mat gradient,int x,int y)
{
	std::vector<double> hist(9, 0);
	int j,i;
	float low ,high ,border;
	float ang , first , second ;
	for(i = x; i <x+6;i++)
		for(j=y;j < y+6; j++)
		{
			ang= angle.at<float>(i,j);
			if(ang <= 10.0)
			hist[0] +=(ang+10)/20 * gradient.at<float>(i,j);
			else if(ang > 10.0 && ang < 170.0)
			{
				low= ang-10;
				high= ang+10;
				int index =(int)ang/20;
				border = ang - (int)ang%20;
				if(low > index*20.0)
				{
					border= border + 20.0;
					index = (int)border/20;
				}
				
				first = (border-low)/20;
				second = (high-border)/20;
				hist[index] +=second*gradient.at<float>(i,j);
				hist[index-1] +=first*gradient.at<float>(i,j);
			}			
			else
			{
				low= 180-ang+10;
				hist[8] += low/20 * gradient.at<float>(i,j); 	
			}  		 
		 }				
	return hist;
}




std::vector<double> getSingleBlockDiscriptor(std::vector< std::vector< std::vector<double> > > cellHistograms,int x,int y)
{
	std::vector<double> block(81, 0);
	int j,i,k,l;
	
	for(i = x,l = 0; i <x+3;i++)
		for(j=y;j < y+3; j++)
			for(k = 0; k < 9; k++)
				block[l++] = cellHistograms[i][j][k];
	return block;
}

Mat gaussianSpatialWindow(Mat img,int blockSize)
{
	Mat gaussiankernel = getGaussianKernel(blockSize, blockSize*0.5, CV_32F);
	mulTransposed(gaussiankernel, gaussiankernel, false);
	Mat imgDiff;
	filter2D( img, imgDiff,-1,gaussiankernel,Point(-1,-1) );
	return imgDiff;
}




std::vector< std::vector< std::vector<double> > > getBlockDiscriptors(std::vector< std::vector< std::vector<double> > > cellHistograms)
{
	std::vector< std::vector< std::vector<double> > > blockDiscriptor;
	std::vector< std::vector<double> > blockDiscriptorRow;
	std::vector<double> blockHist;
	
	for(int i = 0; i < cellHistograms.size(); i+=3)
	{
		for(int j = 0; j < cellHistograms[0].size(); j+=3)
		{
			blockHist = getSingleBlockDiscriptor(cellHistograms,i,j);
			blockDiscriptorRow.push_back(blockHist);
		}
		blockDiscriptor.push_back(blockDiscriptorRow);
	}
	return blockDiscriptor;
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
	magnitude = gaussianSpatialWindow(magnitude,3);
	histogramOfCells(angle,magnitude);
	
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


