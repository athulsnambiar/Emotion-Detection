cmake_minimum_required(VERSION 2.8)
project(FaceDetection)


set(FaceDerection_VERSION_MAJOR 1)
set(FaceDerection_VERSION_MINOR 0)

#SET(CMAKE_CXX_FLAGS_DISTRIBUTION "-O3")
#SET(CMAKE_C_FLAGS_DISTRIBUTION "-O3")
link_directories("${CMAKE_BINARY_DIR}/res")

find_package( OpenCV REQUIRED )
add_executable( annotate.exe annotate.cpp )
target_link_libraries( annotate.exe dlib)
add_executable( train.exe train.cpp )
target_link_libraries( train.exe dlib)
add_executable( emotionRecognition.exe emotionRecognition.cpp )
target_link_libraries( emotionRecognition.exe dlib)
