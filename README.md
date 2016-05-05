# Emotion-Detection

Steps for use

step 01: Download "shape_predictor_68_face_landmarks.dat" for landmark detection from this link http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2

step 02: Put this file to the "data" folder

step 03: Download and setup "dlib" on your computer

step 04: In the programs give proper name to the header files(dlib directory)

step 05: download and install cmake to compile

step 06: For training put images happy and sad images in seperate folders

step 07: run annotate.exe "./annotate.exe (emotion no) (dierctoy of images)/*"

step 08: run ./train.exe

step 09: run ./emotionDetection.exe <image name>
