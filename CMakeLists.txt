cmake_minimum_required(VERSION 2.8)
project(FaceDetection)


set(FaceDerection_VERSION_MAJOR 1)
set(FaceDerection_VERSION_MINOR 0)

include(/home/qburst/opencv3/dlib-18.18/dlib/cmake)

#SET(CMAKE_CXX_FLAGS_DISTRIBUTION "-O3")
#SET(CMAKE_C_FLAGS_DISTRIBUTION "-O3")
link_directories("${CMAKE_BINARY_DIR}/res")

find_package( OpenCV REQUIRED )
add_executable( annotate.exe annotate.cpp )
target_link_libraries(train.exe ${OpenCV_LIBS} dlib)

