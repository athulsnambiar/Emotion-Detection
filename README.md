# Emotion-Detection
##Emotions
* Neutral  0
* Happy    1
* Sad      2
* Anger    3
* Surprise 4
* Disgust  5

##Steps for use

1. Download "shape_predictor_68_face_landmarks.dat" for landmark detection from this link: http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2

2. Put this file to the "data" folder.

3. Download and setup "dlib" on your computer.

4. In the programs give proper name to the header files(dlib directory).

5. Download and install cmake to compile.

6. For training put images containing happy and sad emotions in separate folders.

7. Run annotate.exe "./annotate.exe (emotion no) (directoy of images)/*" .

8. Run ./train.exe .

9. Run ./emotionDetection.exe (image name) .
