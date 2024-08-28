sudo apt install opencl-header ocl-icd-opencl-dev

# bash build.sh
# bash build.sh test_ocl.cpp
#
# source build.sh test_ocl-2-device.cpp # ok
# source build.sh test_ocl-multi-devices.cpp # ok
# source build.sh test_usm_device-multi-devices.cpp # failed
# source build.sh test_ocl-multi-contexts.cpp # failed
# source build.sh test_ocl-re-enter-context.cpp # ok
# source build.sh test_ocl-use-host.cpp # ok

# source build.sh test_ocl-shared-buf.cpp # failed
# source build.sh test_ocl-fine-grain-system-svm.cpp # failed

# compare 2
source build.sh test-ocl_buf_shared.cpp # ok
source build.sh test_ocl-use-host-ts.cpp # ok
./app 2>&1 | tee mylog
