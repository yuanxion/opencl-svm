sudo apt install opencl-header ocl-icd-opencl-dev

# bash build.sh
# bash build.sh test_ocl.cpp
#
# source build.sh test_ocl-2-device.cpp
source build.sh test_ocl-multi-devices.cpp
./app 2>&1 | tee mylog
