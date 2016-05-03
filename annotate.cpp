#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "/home/qburst/opencv3/dlib-18.18/dlib/image_processing/frontal_face_detector.h"
#include "/home/qburst/opencv3/dlib-18.18/dlib/image_processing/render_face_detections.h"
#include "/home/qburst/opencv3/dlib-18.18/dlib/image_processing.h"
#include "/home/qburst/opencv3/dlib-18.18/dlib/gui_widgets.h"
#include "/home/qburst/opencv3/dlib-18.18/dlib/image_io.h"

#include<iostream>
#include<stdio.h>
#include<sstream>
#include<fstream>
#include<cmath>

#define HAPPY		0
#define SAD			1
#define SURPRISE	2
#define ANGRY		3

using namespace std;
using namespace dlib;
using namespace cv;






string shapeFileName = "./data/shape_predictor_68_face_landmarks.dat";
shape_predictor sp;
int faceNumber = 0;

int detectFaceAndCrop(char *imageName);
int getAllAttributes(int noOfFaces,int emotion);
double length(point a,point b);
double slope(point a,point b);


int detectFaceAndCrop(char *imageName)
{
	frontal_face_detector detector = get_frontal_face_detector();
	
	array2d<rgb_pixel> img;
	
	load_image(img,imageName);
	
	pyramid_up(img);
	
	std::vector<dlib::rectangle> faceRectangles = detector(img);
	
	std::vector<full_object_detection> facialFeatures;

	for (int j = 0; j < faceRectangles.size(); ++j)
	{
		full_object_detection feature = sp(img, faceRectangles[j]);
		facialFeatures.push_back(feature);
	}
	
	dlib::array< array2d<rgb_pixel> > faces;
	
	extract_image_chips(img, get_face_chip_details(facialFeatures,500), faces);
	
	for(int i = 0; i < faces.size();i++,faceNumber++)
	{
		stringstream s;
		s<<"face"<<(faceNumber)<<".jpg";
		save_jpeg(faces[i],s.str(),100);
	}
	
	return(faceRectangles.size());	
}

int getAllAttributes(int noOfFaces,int emotion)
{
	int i,j,k;
	frontal_face_detector detector = get_frontal_face_detector();
	
	ofstream outfile;
	ifstream infile("points.csv");
	stringstream s;
	
	outfile.open("points.csv",ios::app);
	
	if(!infile.good())
	{
		for(i = 0; i < 68;i++)
			for(j = 0; j < 68;j++)
				if(i!=j)
					outfile<<"lena"<<i<<"b"<<j<<","<<"a"<<i<<"b"<<j<<",";
		outfile<<"emotion,\n";
	}
	
	for(i = 0; i < faceNumber; i++)
	{
		array2d<rgb_pixel> img;
		s.str("");
		s<<"face"<<(i)<<".jpg";
//		Mat cvimg = imread(s.str(),1);
		load_image(img,s.str());
		
		
		std::vector<dlib::rectangle> faceRectangles = detector(img);
	
		std::vector<full_object_detection> facialFeatures;
		
		full_object_detection feature = sp(img, faceRectangles[0]);
		
		for(int j = 0; j < 68; j++)
			for(int k = 0; k < j; k++)
				outfile<<length(feature.part(j),feature.part(k))<<","<<slope(feature.part(j),feature.part(k))<<",";
		
		outfile<<emotion<<",\n";
	}
	outfile.close();
	infile.close();
	return i;
}


double length(point a,point b)
{
	int x1,y1,x2,y2;
	double dist;	
	x1 = a.x();
	y1 = a.y();
	x2 = b.x();
	y2 = b.y();

	dist = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2);
	dist = sqrt(dist);	
	return dist;
}

double slope(point a,point b)
{
	int x1,y1,x2,y2;

	x1 = a.x();
	y1 = a.y();
	x2 = b.x();
	y2 = b.y();
	if((x1-x2) == 0)
		if((y1-y2) > 0)
			return (M_PI/2);
		else
			return (-M_PI/2);
	else
		return atan(double(y1-y2))/(x1-x2);
}

int main(int argc,char **argv)
{
	int noOfFaces = 0;
	if(argc < 3)
	{
		printf("\n\n!!Arguments not Given properly!!\n\n");
	}
	deserialize(shapeFileName) >> sp;
	cout<<"\n\nProgram Started\n\n";
	for(int i = 2;i < argc; i++)
		noOfFaces += detectFaceAndCrop(argv[i]);
	//getEyes(noOfFaces,(int)(argv[1][0]-'0'));
	getAllAttributes(noOfFaces,(int)(argv[1][0]-'0'));
	return 0;
}

