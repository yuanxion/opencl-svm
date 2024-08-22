#!/usr/bin/bash

target_file=$1

#g++ test_cv_ocl_svm.cpp -o app -std=c++17 -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
#g++ test_ocl_svm.cpp -o app -std=c++17 -I/usr/local/include/opencv4 -L/usr/local/lib -lOpenCL

g++ $target_file -o app -std=c++17 -I/usr/local/include/opencv4 -L/usr/local/lib -lOpenCL

