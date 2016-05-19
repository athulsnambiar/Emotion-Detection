#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cmath>
#include <stdlib.h>

using namespace std;
using namespace cv;

typedef std::vector<double> vector1D;
typedef std::vector< std::vector<double> > vector2D;
typedef std::vector< std::vector< std::vector<double> > > vector3D;



bool isPixelInside(int row,int col,int x,int y);

Mat diffsizekernel(Mat img, int f, int c);

Mat diffx(Mat img);

Mat diffy(Mat img);

string type2str(int type);

Mat remRowCol(Mat img,int r,int c);

Mat addPi(Mat angle);

Mat gaussianSpatialWindow(Mat img,int blockSize = 3);

vector3D histogramOfCells(Mat angle,Mat gradient,int cellSize = 6,int binSize = 9);

vector1D binning(Mat angle,Mat gradient,int x,int y,int cellSize = 6,int binSize=9);

vector3D getBlockDescriptors(vector3D cellHistograms,int blockSize = 3,int binSize = 9);

vector1D getSingleBlockDescriptor(vector3D cellHistograms,int x,int y,int blockSize = 3);

vector3D normalizeBlockDescriptor(vector3D blockDescriptor,int normalizeType = 0,double threshold = 0.2,double e = 0.0);

vector1D l2Hys(vector1D array,double threshold = 0.2,double e = 0.0);

vector1D l2Norm(vector1D array,double e = 0.0);

vector1D l1Sqrt(vector1D array,double e = 0.0);

vector3D getHOGFeature(Mat image,int cellSize = 6,int binSize = 9,int blockSize = 3,int normType  = 0,double normThreshold = 0.2,double normE = 0.0);

vector3D readImage(int argc,char **argv);

void writeHOGFile(vector3D HOG);

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

vector3D histogramOfCells(Mat angle,Mat gradient,int cellSize,int binSize)
{
	vector3D cellHist;
	vector2D cellHistRow;
	vector1D hist;
	int col = angle.cols;
	int row = angle.rows;
	int i,j;
	for(i = 0; i <row;i += cellSize)
	{
		cellHistRow.clear();
		for(j=0;j < col; j += cellSize)
		{
			hist = binning(angle,gradient,i,j,cellSize,binSize);
			cellHistRow.push_back(hist);
		}
		cellHist.push_back(cellHistRow);
	}
	return cellHist;


}

vector1D binning(Mat angle,Mat gradient,int x,int y , int cellSize, int binSize)
{
	vector1D hist(binSize, 0);
	int j,i;
	float low ,high ,border,span,hspan;
	float ang , first , second ;
	span = 180.0/binSize;
	hspan = span/2;
	for(i = x; i <x+cellSize;i++)
		for(j=y;j < y+cellSize; j++)
		{
			ang= angle.at<float>(i,j);
			if(ang <= hspan)
			 hist[0] +=(ang+hspan)/span * gradient.at<float>(i,j);
			else if(ang > hspan && ang < (180 - hspan))
			{
				low= ang-hspan;
				high= ang+hspan;
				int index =(int)ang/span;
				border = ang - (int)ang%(int)span;
				if(low > index*span)
				{
					border= border + span;
					index = (int)border/span;
				}

				first = (border-low)/span;
				second = (high-border)/span;
				hist[index] +=second*gradient.at<float>(i,j);
				hist[index-1] +=first*gradient.at<float>(i,j);
			}
			else
			{
				low= 180-ang+hspan;
				hist[binSize-1] += low/span * gradient.at<float>(i,j);
			}
		 }
	return hist;
}





vector1D getSingleBlockDescriptor(vector3D cellHistograms,int x,int y,int blockSize)
{
	vector1D block;
	int j,i,k,l;

	for(i = x,l = 0; i <x+blockSize;i++)
		for(j = y;j < y+blockSize; j++)
			block.insert(block.end(),cellHistograms[i][j].begin(),cellHistograms[i][j].end());
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




vector3D getBlockDescriptors(vector3D cellHistograms,int blockSize,int binSize)
{
	vector3D blockDescriptor;
	vector2D blockDescriptorRow;
	vector1D blockHist;

	for(int i = 0; i < cellHistograms.size()-blockSize+1; i++)
	{
		blockDescriptorRow.clear();
		for(int j = 0; j < cellHistograms[0].size()-blockSize+1; j++)
		{
			blockHist = getSingleBlockDescriptor(cellHistograms,i,j,blockSize);
			blockDescriptorRow.push_back(blockHist);
		}
		blockDescriptor.push_back(blockDescriptorRow);
	}
	return blockDescriptor;
}


vector3D normalizeBlockDescriptor(vector3D blockDescriptor,int normalizeType,double threshold,double e)
{
	int i,j;

	if(normalizeType != 1 && normalizeType != 2 )
	{
		for(i = 0; i < blockDescriptor.size(); i++)
		{
			for(j = 0; j < blockDescriptor[0].size(); j++)
			{
				blockDescriptor[i][j] = l2Norm(blockDescriptor[i][j],e);
			}
		}
	}

	else if(normalizeType == 1 )
	{
		for(i = 0; i < blockDescriptor.size(); i++)
		{
			for(j = 0; j < blockDescriptor[0].size(); j++)
			{
				blockDescriptor[i][j] = l2Hys(blockDescriptor[i][j],threshold,e);
			}
		}
	}

	else
	{
		for(i = 0; i < blockDescriptor.size(); i++)
		{
			for(j = 0; j < blockDescriptor[0].size(); j++)
			{
				blockDescriptor[i][j] = l1Sqrt(blockDescriptor[i][j],e);
			}
		}
	}

	return blockDescriptor;
}


vector1D l2Hys(vector1D array,double threshold,double e)
{
	int length = array.size();
	array = l2Norm(array,e);
	for(int i = 0; i < length; i++)
	{
		if(array[i] >= 0.2)
			array[i] = 0.2;
	}
	array = l2Norm(array,e);
	return array;
}


vector1D l2Norm(vector1D array,double e)
{
	double sum = 0;
	int length = array.size();
	for(int i = 0; i < length; i++)
	{
		sum += array[i]*array[i];
	}
	sum += e*e;
	sum = sqrt(sum);

	for(int i = 0; i < length; i++)
	{
		array[i] /= sum;
	}

	return array;
}


vector1D l1Sqrt(vector1D array,double e)
{
	double sum = 0;
	int length = array.size();
	for(int i = 0; i < length; i++)
	{
		sum += array[i];
	}
	sum += e;


	for(int i = 0; i < length; i++)
	{
		array[i] = sqrt(array[i]/sum);
	}

	return array;
}



vector3D getHOGFeature(Mat image,int cellSize,int binSize,int blockSize,int normType,double normThreshold,double normE)
{
	vector3D HOG,cellHist;


	if(image.rows%cellSize != 0 || image.cols%cellSize != 0)
		image = remRowCol(image,image.rows%cellSize,image.cols%cellSize);

	Mat x,y;
	x = diffx(image);
	y = diffy(image);

	Mat magnitude(x.size(),x.type());
	Mat angle(x.size(),x.type());

	cartToPolar(x,y,magnitude,angle,true);

	magnitude = gaussianSpatialWindow(magnitude,blockSize);

	angle = addPi(angle);

	cellHist = histogramOfCells(angle,magnitude,cellSize,binSize);

	HOG = getBlockDescriptors(cellHist,blockSize,binSize);

	HOG = normalizeBlockDescriptor(HOG,normType,normThreshold,normE);

	return HOG;
}



vector3D readImage(int argc,char **argv)
{
	int cellSize = 6;
	int binSize = 9;
	int blockSize = 3;
	int normType = 0;
	double normThreshold = 0.2;
	double normE = 0.0;
	vector3D HOGimage;
	Mat image,img32;


	for(int i = 1; i < argc;i++)
	{
		cout<<"\nimage = "<<i<<"\t"<<argv[i];
		image = imread(argv[i],CV_LOAD_IMAGE_GRAYSCALE);
		if(image.rows == 0)
			continue;
		image.convertTo(img32,CV_32F);
		HOGimage = getHOGFeature(img32,cellSize,binSize,blockSize,normType,normThreshold,normE);
		cout<<"\tdone\n\n";
	//writeHOGFile(HOGimage);
	}
	return HOGimage;
}

void writeHOGFile(vector3D HOG)
{
	ofstream file("hog.csv");
	int i,j,k;
	for(i = 0; i < HOG.size();i++)
		for(j = 0; j < HOG[0].size();j++)
			for(k = 0; k < HOG[0][0].size();k++)
				file <<HOG[i][j][k]<<"\n";
	file.close();
}

int main(int argc,char **argv)
{
	if(argc < 2)
	{
		cout<<"No input Image";
		return 0;
	}

	readImage(argc,argv);
	return 0;
}
