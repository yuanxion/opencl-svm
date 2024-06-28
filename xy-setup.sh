sudo apt install opencl-header ocl-icd-opencl-dev

bash build.sh
bash build.sh test_ocl.cpp

./app 2>&1 | tee mylog
