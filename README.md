# Emotion-Detection

Steps for use

Step 01: Download "shape_predictor_68_face_landmarks.dat" for landmark detection from this link: http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2

Step 02: Put this file to the "data" folder.

Step 03: Download and setup "dlib" on your computer.

Step 04: In the programs give proper name to the header files(dlib directory).

Step 05: Download and install cmake to compile.

Step 06: For training put images containing happy and sad emotions in separate folders.

Step 07: Run annotate.exe "./annotate.exe (emotion no) (directoy of images)/*" .

Step 08: Run ./train.exe .

Step 09: Run ./emotionDetection.exe (image name) .
