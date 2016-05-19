#include "/home/qburst/opencv3/dlib-18.18/dlib/image_processing/frontal_face_detector.h"
#include "/home/qburst/opencv3/dlib-18.18/dlib/image_processing.h"
#include "/home/qburst/opencv3/dlib-18.18/dlib/image_io.h"

#include<iostream>
#include<stdio.h>
#include<sstream>
#include<fstream>
#include<cmath>
#include<vector>
#include<cstdio>

using namespace std;
using namespace dlib;


typedef matrix<double,4556,1> sample_type;
typedef radial_basis_kernel<sample_type> kernel_type;
typedef probabilistic_decision_function<kernel_type> probabilistic_funct_type;
typedef normalized_function<probabilistic_funct_type> pfunct_type;




string emotionFileName = "emotion_predictor_data.dat";
string shapeFileName = "./data/shape_predictor_68_face_landmarks.dat";
shape_predictor sp;
pfunct_type ep;
int faceNumber = 0;




int detectFaceAndCrop(char *imageName);
std::vector<sample_type> getAllAttributes(int noOfFaces);
double length(point a,point b);
double slope(point a,point b);
void removePhotos();




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
	std::vector<dlib::rectangle> faceCheck;
	for(int i = 0; i < faces.size();i++)
	{
		stringstream s;
		faceCheck = detector(faces[i]);
		if(faceCheck.size() <= 0)
			continue;
		s<<"face"<<(faceNumber++)<<".jpg";
		save_jpeg(faces[i],s.str(),100);
	}

	return(faceRectangles.size());
}




std::vector<sample_type> getAllAttributes(int noOfFaces)
{
	int i,j,k;
	frontal_face_detector detector = get_frontal_face_detector();
	std::vector<sample_type> samples;
	sample_type sample;
	stringstream s;


	for(i = 0; i < faceNumber; i++)
	{
		array2d<rgb_pixel> img;
		s.str("");
		s<<"face"<<(i)<<".jpg";
		load_image(img,s.str());


		std::vector<dlib::rectangle> faceRectangles = detector(img);

		std::vector<full_object_detection> facialFeatures;

		full_object_detection feature = sp(img, faceRectangles[0]);
		int l = 0;
		for(int j = 0; j < 68; j++)
			for(int k = 0; k < j; k++,l++)
			{
				sample(l) = length(feature.part(j),feature.part(k));
				l++;
				sample(l) = slope(feature.part(j),feature.part(k));

			}
		samples.push_back(sample);

	}
	return samples;
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




void removePhotos()
{
	int i;
	stringstream s;

	for(i = 0; i < faceNumber; i++)
	{
		s.str("");
		s << "face" << i << ".jpg";
		remove(s.str().c_str());
	}
}








int main(int argc,char **argv)
{
	int noOfFaces = 0;
	std::vector<sample_type> samples;
	if(argc < 2)
	{
		printf("\n\n!!Arguments not Given properly!!\n\n");
	}
	deserialize(shapeFileName) >> sp;
	deserialize(emotionFileName) >> ep;


	cout<<"\n\nProgram Started\n\n";

	noOfFaces += detectFaceAndCrop(argv[1]);

	samples = getAllAttributes(noOfFaces);
	for(int i = 0; i < faceNumber; i++)
	{
		cout << "probablity that face "<<i<<" is happy :" << ep(samples[i]) << endl;
		cout << "probablity that face "<<i<<" is sad   :" << (1.0 - ep(samples[i])) << "\n\n\n";
	}
	cout<<"\n\nPress Enter to delete all Photos.............";
	cin.ignore();
	removePhotos();
	return 0;

}
