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
#include<vector>

using namespace std;
using namespace dlib;
using namespace cv;


typedef matrix<double,4556,1> sample_type;

std::vector<std::vector <float> > getAttributesCSV(char * name);
bool rowsAndCols(char *name,int &row,int &col);
std::vector <int> getLabelsCSV(char * name);
void generateData(std::vector<sample_type>& samples,std::vector<double>& labels);



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
			stringstream cell1(cell);
			cell1 >> cellValue;
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


void generateData(std::vector<sample_type>& samples,std::vector<double>& labels)
{
		std::vector<std::vector <float> > matrix = getAttributesCSV("points.csv");
	std::vector <int> matLabel = getLabelsCSV("points.csv");
	sample_type temp;
	
	
	for(int i = 0; i < matrix.size();i++)
	{
		for(int j = 0; j < matrix[0].size();j++)
		{
			temp(j) = matrix[i][j];
		}
		samples.push_back(temp);
		if(matLabel[i] == 0)
			labels.push_back(+1);
		if(matLabel[i] == 1)
			labels.push_back(-1);
	}
}

int main()
{
	try
	{
		std::vector<sample_type> samples;
		std::vector<double> labels;
		
		generateData(samples, labels);

		cout << "samples.size(): "<< samples.size() << endl;
		
		typedef radial_basis_kernel<sample_type> kernel_type;
		
		vector_normalizer<sample_type> normalizer;
		
		normalizer.train(samples);
		
		for (unsigned long i = 0; i < samples.size(); ++i)
		samples[i] = normalizer(samples[i]);
		
		randomize_samples(samples, labels);
		
		const double max_nu = maximum_nu(labels);
		
		svm_nu_trainer<kernel_type> trainer;
		
		cout << "doing cross validation" << endl;
		for (double gamma = 0.00001; gamma <= 1; gamma *= 5)
		{
			for (double nu = 0.00001; nu < max_nu; nu *= 5)
			{
				trainer.set_kernel(kernel_type(gamma));
				trainer.set_nu(nu);
	
				cout << "gamma: " << gamma << "	nu: " << nu;
	
				cout << "	 cross validation accuracy: " << cross_validate_trainer(trainer, samples, labels, 3);
			}
		}
		
	trainer.set_kernel(kernel_type(1e-5));
	trainer.set_nu(0.03125);
	typedef decision_function<kernel_type> dec_funct_type;
	typedef normalized_function<dec_funct_type> funct_type;
	
	funct_type learned_function;
	learned_function.normalizer = normalizer;
	learned_function.function = trainer.train(samples, labels);
	
	cout << "\nnumber of support vectors in our learned_function is " 
		 << learned_function.function.basis_vectors.size() << endl;
		 
	typedef probabilistic_decision_function<kernel_type> probabilistic_funct_type;  
	typedef normalized_function<probabilistic_funct_type> pfunct_type;

	pfunct_type learned_pfunct; 
	learned_pfunct.normalizer = normalizer;
	learned_pfunct.function = train_probabilistic_decision_function(trainer, samples, labels, 3);
	
	cout << "\nnumber of support vectors in our learned_pfunct is "<< learned_pfunct.function.decision_funct.basis_vectors.size() << endl;
	serialize("emotion_predictor_data.dat") << learned_pfunct;
	
		cout << "\ncross validation accuracy with only 10 support vectors: " 
		 << cross_validate_trainer(reduced2(trainer,10), samples, labels, 3);

	cout << "cross validation accuracy with all the original support vectors: " 
		 << cross_validate_trainer(trainer, samples, labels, 3);



	learned_function.function = reduced2(trainer,10).train(samples, labels);
	learned_pfunct.function = train_probabilistic_decision_function(reduced2(trainer,10), samples, labels, 3);
	}
		
	catch (std::exception& e)
	{
		cout << "exception thrown!" << endl;
		cout << e.what() << endl;
	}
}
