//#include <CL/cl.hpp>
#include <CL/cl2.hpp>

#include <iostream>
#include <vector>

const char* kernelSource = R"(
    __kernel void vectorAdd(__global int* a, __global int* b, __global int* c, int n) {
        int i = get_global_id(0);
        printf("id: %d, a,b: %d,%d\n", i, a[i],b[i]);
        if (i < n) {
            c[i] = a[i] + b[i];
            //printf("id: %d, a,b,c: %d,%d,%d\n", i, a[i],b[i],c[i]);
            //c[i] = 0;
        }
    }
)";

void printDeviceInfo(cl::Device& device) {
    std::cout << "### Device ### "  << std::endl;

    std::cout << "Device Name: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    std::cout << "Device Vendor: " << device.getInfo<CL_DEVICE_VENDOR>() << std::endl;
    std::cout << "Device Version: " << device.getInfo<CL_DEVICE_VERSION>() << std::endl;
    std::cout << "Driver Version: " << device.getInfo<CL_DRIVER_VERSION>() << std::endl;
    std::cout << "OpenCL C Version: " << device.getInfo<CL_DEVICE_OPENCL_C_VERSION>() << std::endl;
    
    cl_ulong globalMemSize = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    std::cout << "Global Memory Size: " << globalMemSize / (1024 * 1024) << " MB" << std::endl;
    
    cl_ulong localMemSize = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
    std::cout << "Local Memory Size: " << localMemSize / 1024 << " KB" << std::endl;
    
    cl_uint computeUnits = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    std::cout << "Compute Units: " << computeUnits << std::endl;
    
    size_t workGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    std::cout << "Max Work Group Size: " << workGroupSize << std::endl;
    std::cout << "### ### "  << std::endl;
}

int main() {
    // 初始化OpenCL环境
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms[0];
    //cl::Platform platform = platforms[1];

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices[0];
    //cl::Device device = devices[1];

    printDeviceInfo(device);

    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    // 检查设备是否支持SVM
    cl_device_svm_capabilities caps;
    clGetDeviceInfo(device(), CL_DEVICE_SVM_CAPABILITIES, sizeof(caps), &caps, nullptr);

    bool supported = false;
    // 对应粗粒度缓冲SVM，细粒度缓冲SVM，细粒度系统SVM
    if ((caps & CL_DEVICE_SVM_ATOMICS)) {
        supported = true;
        std::cout << "Device support atomic SVM." << std::endl;
    }
    if ((caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM)) {
        supported = true;
        std::cout << "Device support fine-grained system SVM." << std::endl;
    }
    if ((caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)) {
        supported = true;
        std::cout << "Device support fine-grained buffer SVM." << std::endl;
    }
    if ((caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)) {
        supported = true;
        std::cout << "Device support coarse-grained buffer SVM." << std::endl;
    }
    if (!supported) {
        std::cout << "Device does not support any SVM." << std::endl;
        return -1;
    }

    // 创建程序和内核
    cl::Program program(context, kernelSource);
    program.build({device});
    cl::Kernel kernel(program, "vectorAdd");
    std::cout << "Kernel created" << std::endl;

    std::cout << "分配SVM内存" << std::endl;
    const int arraySize = 16; // 32; //  256; // 1024;
    float* a = (float*)aligned_alloc(arraySize, sizeof(cl_float16));
    float* b = (float*)aligned_alloc(arraySize, sizeof(cl_float16));
    float* c = (float*)aligned_alloc(arraySize, sizeof(cl_float16));

    // 初始化数据
    std::cout << "初始化数据" << std::endl;
    for (int i = 0; i < arraySize; i++) {
        a[i] = i;
        b[i] = i * 10;
        c[i] = -1;
        std::cout << "i: " << i << " a,b,c: " << a[i] << "," << b[i] << "," << c[i] << std::endl;
    }

    // 设置内核参数
    std::cout << "设置内核参数" << std::endl;
    // kernel.setArg(0, a);
    // kernel.setArg(1, b);
    // kernel.setArg(2, c);
    // kernel.setArg(3, arraySize);
    cl_int err;
    err = clSetKernelArgSVMPointer(kernel(), 0, a);
    if (err != CL_SUCCESS) {
        std::cerr << "设置内核参数失败，错误代码: " << err << std::endl;
        return -1;
    }
    clSetKernelArgSVMPointer(kernel(), 1, b);
    clSetKernelArgSVMPointer(kernel(), 2, c);
    clSetKernelArg(kernel(), 3, sizeof(int), &arraySize);

    // double check
    for (int i = 0; i < arraySize; i++) {
        std::cout << "check i: " << i << " a,b,c: " << a[i] << "," << b[i] << "," << c[i] << std::endl;
    }

    // 执行内核
    std::cout << "执行内核" << std::endl;
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(arraySize));
    queue.finish();

    // 验证结果
    for (int i = 0; i < arraySize; i++) {
        std::cout << "i: " << i << ", Expected: " << i+i*10 << ", got: " << c[i] << std::endl;
        // if (c[i] != i + i * 10) {
        //     std::cout << "Verification failed at index " << i << std::endl;
        //     break;
        // }
    }

    std::cout << "Computation completed successfully." << std::endl;
    return 0;
}
