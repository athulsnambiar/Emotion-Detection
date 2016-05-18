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


string emotionFileName1 = "neutral_vs_happy.dat";
string emotionFileName2 = "neutral_vs_sad.dat";
string emotionFileName3 = "neutral_vs_surprise.dat";
string emotionFileName4 = "happy_vs_sad.dat";
string emotionFileName5 = "happy_vs_surprise.dat";
string emotionFileName6 = "sad_vs_surprise.dat";
string shapeFileName = "./data/shape_predictor_68_face_landmarks.dat";
shape_predictor sp;
pfunct_type ep1;
pfunct_type ep2;
pfunct_type ep3;
pfunct_type ep4;
pfunct_type ep5;
pfunct_type ep6;
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

	for(int i = 0; i < faces.size();i++,faceNumber++)
	{
		stringstream s;
		s<<"face"<<(faceNumber)<<".jpg";
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


std::vector<double> svmMulticlass(sample_type sample)
{
	std::vector<double> probs;
	probs.push_back(ep1(sample));
	probs.push_back(ep2(sample));
	probs.push_back(ep3(sample));
	probs.push_back(ep4(sample));
	probs.push_back(ep5(sample));
	probs.push_back(ep6(sample));

	return probs;
}

int main(int argc,char** argv)
{
	int noOfFaces = 0;
	std::vector<sample_type> samples;
	if(argc < 2)
	{
		printf("\n\n!!Arguments not Given properly!!\n\n");
	}
	deserialize(shapeFileName) >> sp;
	deserialize(emotionFileName1) >> ep1;
	deserialize(emotionFileName2) >> ep2;
	deserialize(emotionFileName3) >> ep3;
	deserialize(emotionFileName4) >> ep4;
	deserialize(emotionFileName5) >> ep5;
	deserialize(emotionFileName6) >> ep6;

	cout<<"\n\nProgram Started\n\n";

	noOfFaces += detectFaceAndCrop(argv[1]);

	samples = getAllAttributes(noOfFaces);

	for(int i = 0; i < faceNumber; i++)
	{
		std::vector<double> prob;
		prob = svmMulticlass(samples[i]);
		cout << "probablity that face "<<i<<" is Neutral  :" << prob[0] << endl;
		cout << "probablity that face "<<i<<" is Happy    :" << prob[0] << endl;
		cout << "probablity that face "<<i<<" is Sad      :" << prob[0] << endl;
		cout << "probablity that face "<<i<<" is Surprise :" << prob[0] << "\n\n\n";
	}

	cout<<"\n\nPress Enter to delete all Photos.............";
	cin.ignore();
	removePhotos();
}

