#include "/home/qburst/opencv3/dlib-18.18/dlib/image_processing/frontal_face_detector.h"
#include "/home/qburst/opencv3/dlib-18.18/dlib/image_processing.h"
#include "/home/qburst/opencv3/dlib-18.18/dlib/image_io.h"
#include "/home/qburst/opencv3/dlib-18.18/dlib/svm_threaded.h"

#include<iostream>
#include<stdio.h>
#include<sstream>
#include<fstream>
#include<cmath>
#include<vector>

using namespace std;
using namespace dlib;


typedef matrix<double,4556,1> sample_type;

std::vector<std::vector <float> > getAttributesCSV(char * name);
bool rowsAndCols(char *name,int &row,int &col);
std::vector <int> getLabelsCSV(char * name);
void generateData(std::vector <std::vector<sample_type> >& samples);
void trainEmotion(std::vector <std::vector<sample_type> > samplesSet);
void train2Emotion(std::vector<sample_type> sample1,std::vector<sample_type> sample2,string filename);



bool rowsAndCols(char *name,int &row,int &col)
{
	ifstream file(name);

	if (!file)
	{
		cout << "can not open file" << endl;
		row = col = -1;
		return false;
	}
	else
	{
		row = col = 0;
		char c;
		while (c = file.get())
		{
			if(!file.good())
				break;
			if(c == '\n')
				row++,col++;
			else if(c == ',')
				col++;
		}

		col = col/row;
		file.close();
		return true;
	}
}

std::vector <int> getLabelsCSV(char * name)
{

	std::ifstream  file(name);
	int cellValue;
	int row,col,r,c;
	rowsAndCols(name,row,col);
	std::string line;
	std::vector<int> labels;

	for(r = 0; r < row; r++)
	{
		getline(file,line);
		stringstream lineStream(line);
		string cell;
		if(r == 0)
			continue;
		for(c = 0; c < col; c++)
		{

			std::getline(lineStream,cell,',');
			if(c != col-1)
				continue;
			if(cell.compare("neutral") == 0)
				cellValue = 0;
			else if(cell.compare("happy") == 0)
				cellValue = 1;
			else if(cell.compare("sad") == 0)
				cellValue = 2;
			else if(cell.compare("surprise") == 0)
				cellValue = 3;
			labels.push_back(cellValue);
		}
	}
	file.close();
	return labels;
}


std::vector<std::vector <float> > getAttributesCSV(char * name)
{
	std::ifstream  file(name);
	float cellValue;
	int row,col,r,c;
	rowsAndCols(name,row,col);
	std::string line;
	std::vector< std::vector <float> > Matrix;

	for(r = 0; r < row; r++)
	{
		getline(file,line);
		stringstream lineStream(line);
		std::vector <float> row1;
		string cell;
		if(r == 0)
			continue;
		for(c = 0; c < col; c++)
		{
			if(c == col-1)
				continue;
			std::getline(lineStream,cell,',');
			stringstream cell1(cell);
			cell1 >> cellValue;
			row1.push_back(cellValue);
		}
		Matrix.push_back(row1);
	}
	file.close();
	return Matrix;
}


void generateData(std::vector <std::vector<sample_type> >& samplesSet)
{
	char filename[] = "points.csv";
	std::vector<std::vector <float> > matrix = getAttributesCSV(filename);
	std::vector <int> matLabel = getLabelsCSV(filename);
	std::vector<sample_type> neutral,happy,sad,surprise;
	sample_type temp;
	int h = 0,sa = 0,su = 0,n = 0;

	for(int i = 0; i < matrix.size();i++)
	{
		for(int j = 0; j < matrix[0].size();j++)
		{
			temp(j) = matrix[i][j];
		}

		if(matLabel[i] == 0)
			neutral.push_back(temp);
		else if(matLabel[i] == 1)
			happy.push_back(temp);
		else if(matLabel[i] == 2)
			sad.push_back(temp);
		else if(matLabel[i] == 3)
			surprise.push_back(temp);
	}
	cout<<"Neutral  :" <<neutral.size()<<"\n";
	cout<<"Happy    :" <<happy.size()<<"\n";
	cout<<"Sad      :" <<sad.size()<<"\n";
	cout<<"Surprise :" <<surprise.size()<<"\n";
	samplesSet.push_back(neutral);
	samplesSet.push_back(happy);
	samplesSet.push_back(sad);
	samplesSet.push_back(surprise);

}

void train2Emotion(std::vector<sample_type> sample1,std::vector<sample_type> sample2,string filename)
{
	std::vector<sample_type> samples;
	std::vector<double> labels;
	std::vector<double> lab1(sample1.size(),1),lab2(sample2.size(),-1);
	samples.insert(samples.end(),sample1.begin(),sample1.end());
	samples.insert(samples.end(),sample2.begin(),sample2.end());
	labels.insert(labels.end(),lab1.begin(),lab1.end());
	labels.insert(labels.end(),lab2.begin(),lab2.end());

	cout << "samples.size(): "<< samples.size() << endl;
	cout << "labels.size() : "<< labels.size() << endl;

	typedef radial_basis_kernel<sample_type> kernel_type;

	vector_normalizer<sample_type> normalizer;

	normalizer.train(samples);

	for (unsigned long i = 0; i < samples.size(); ++i)
	samples[i] = normalizer(samples[i]);

	randomize_samples(samples, labels);
	svm_nu_trainer<kernel_type> trainer;

	trainer.set_kernel(kernel_type(1.4641e-05));
	trainer.set_nu(0.0498789);

	typedef probabilistic_decision_function<kernel_type> probabilistic_funct_type;
	typedef normalized_function<probabilistic_funct_type> pfunct_type;

	pfunct_type learned_pfunct;
	learned_pfunct.normalizer = normalizer;
	learned_pfunct.function = train_probabilistic_decision_function(trainer, samples, labels, 3);

	serialize(filename) << learned_pfunct;
}

void trainEmotion(std::vector <std::vector<sample_type> > sampleSet)
{
	train2Emotion(sampleSet[0],sampleSet[1],"neutral_vs_happy.dat");
	train2Emotion(sampleSet[0],sampleSet[2],"neutral_vs_sad.dat");
	train2Emotion(sampleSet[0],sampleSet[3],"neutral_vs_surprise.dat");
	train2Emotion(sampleSet[1],sampleSet[2],"happy_vs_sad.dat");
	train2Emotion(sampleSet[1],sampleSet[3],"happy_vs_surprise.dat");
	train2Emotion(sampleSet[2],sampleSet[3],"sad_vs_surprise.dat");
}



int main()
{
	std::vector <std::vector<sample_type> > samplesSet;
	std::vector<double> labels;

	generateData(samplesSet);
	trainEmotion(samplesSet);
}
