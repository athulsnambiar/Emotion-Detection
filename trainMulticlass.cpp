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
		labels.push_back(matLabel[i]);
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
		
		typedef one_vs_one_trainer<any_trainer<sample_type> > ovo_trainer;
		
		ovo_trainer trainer;
		
		typedef radial_basis_kernel<sample_type> rbf_kernel;
		
		svm_nu_trainer<rbf_kernel> rbf_trainer;
		
		rbf_trainer.set_kernel(rbf_kernel(1.4641e-05));
		rbf_trainer.set_nu(0.0498789);
		
		cout << "Trainer: "<< samples.size() << endl;
		trainer.set_trainer(rbf_trainer);
		cout << "RBF trainer Set\n\n";
		cout << "Randomizing Samples set\n\n";
		randomize_samples(samples, labels);
		
		//dont cross validate if the database is large. process will run out of heap space
		//cout << "cross validation: \n" << cross_validate_multiclass_trainer(trainer, samples, labels, 3) << endl;
		
		cout << "Creating One vs One Training Function\n\n";
		one_vs_one_decision_function<ovo_trainer> df = trainer.train(samples, labels);
		
		one_vs_one_decision_function<ovo_trainer, decision_function<rbf_kernel> > df2;
		cout << "Preparing to write on disk\n\n";
		df2 = df;
		cout << "Writing training Data on disk\n\n";
		serialize("multiple_emotion_data.dat") << df2;
		
	}
	catch (std::exception& e)
	{
		cout << "exception thrown!" << endl;
		cout << e.what() << endl;
	}
		
}


