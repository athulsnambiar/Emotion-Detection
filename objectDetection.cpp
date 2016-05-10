#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

int main(int argc,char **argv)
{
	float mask[] = {-1,0,1};
	Mat image = imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
	
	namedWindow( "Display window", WINDOW_AUTOSIZE );
    imshow( "Display window", image );
	
	waitKey(0);
	destroyWindow("Display window");
    return 0;
	
}
