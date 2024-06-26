//#include <CL/cl.hpp>
#include <CL/cl2.hpp>

#include <iostream>
#include <vector>

const char* kernelSource = R"(
    __kernel void vectorAdd(__global int* a, __global int* b, __global int* c, int n) {
        int i = get_global_id(0);
        //printf("id: %d\n", i);
        if (i < n) {
            c[i] = a[i] + b[i];
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

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices[0];

    printDeviceInfo(device);

    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    // 检查设备是否支持SVM
    cl_device_svm_capabilities caps;
    clGetDeviceInfo(device(), CL_DEVICE_SVM_CAPABILITIES, sizeof(caps), &caps, nullptr);

    bool supported = false;
    // 对应粗粒度缓冲SVM，细粒度缓冲SVM，细粒度系统SVM
    if ((caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)) {
        supported = true;
        std::cout << "Device support coarse-grained buffer SVM." << std::endl;
    }
    if ((caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)) {
        supported = true;
        std::cout << "Device support fine-grained buffer SVM." << std::endl;
    }
    if ((caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM)) {
        supported = true;
        std::cout << "Device support fine-grained system SVM." << std::endl;
    }
    if ((caps & CL_DEVICE_SVM_ATOMICS)) {
        supported = true;
        std::cout << "Device support atomic SVM." << std::endl;
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

    // 分配SVM内存
    const int arraySize = 1000;
    int* a = (int*)clSVMAlloc(context(), CL_MEM_READ_WRITE, sizeof(int) * arraySize, 0);
    int* b = (int*)clSVMAlloc(context(), CL_MEM_READ_WRITE, sizeof(int) * arraySize, 0);
    int* c = (int*)clSVMAlloc(context(), CL_MEM_READ_WRITE, sizeof(int) * arraySize, 0);

    // 映射SVM内存:写
    cl_int err = clEnqueueSVMMap(queue(), CL_TRUE, CL_MAP_WRITE, a, arraySize, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Error mapping SVM memory a: " << err << std::endl;
        return -1;
    }
    err = clEnqueueSVMMap(queue(), CL_TRUE, CL_MAP_WRITE, b, arraySize, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Error mapping SVM memory b: " << err << std::endl;
        return -1;
    }
    err = clEnqueueSVMMap(queue(), CL_TRUE, CL_MAP_WRITE, c, arraySize, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Error mapping SVM memory c: " << err << std::endl;
        return -1;
    }

    // 初始化数据
    for (int i = 0; i < arraySize; i++) {
        a[i] = i;
        b[i] = i * 2;
        c[i] = 0;
    }

    // 取消映射SVM内存
    err = clEnqueueSVMUnmap(queue(), a, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Error unmapping SVM memory a: " << err << std::endl;
        return -1;
    }
    err = clEnqueueSVMUnmap(queue(), b, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Error unmapping SVM memory b: " << err << std::endl;
        return -1;
    }
    err = clEnqueueSVMUnmap(queue(), c, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Error unmapping SVM memory c: " << err << std::endl;
        return -1;
    }


    // 设置内核参数
    // kernel.setArg(0, a);
    // kernel.setArg(1, b);
    // kernel.setArg(2, c);
    // kernel.setArg(3, arraySize);

    //kernel.setSVMPointers(0, a);
    //kernel.setSVMPointers(1, b);
    //kernel.setSVMPointers(2, c);
    //kernel.setArg(3, arraySize);

    clSetKernelArgSVMPointer(kernel(), 0, a);
    clSetKernelArgSVMPointer(kernel(), 1, b);
    clSetKernelArgSVMPointer(kernel(), 2, c);
    clSetKernelArg(kernel(), 3, sizeof(int), &arraySize);

    // 执行内核
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(arraySize));
    queue.finish();

    // 映射SVM内存:读
    err = clEnqueueSVMMap(queue(), CL_TRUE, CL_MAP_READ, c, arraySize, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Error mapping SVM memory c: " << err << std::endl;
        return -1;
    }

    // 验证结果
    for (int i = 0; i < arraySize; i++) {
        std::cout << "Expected: " << i+i*2 << ", got: " << c[i] << std::endl;
        if (c[i] != i + i * 2) {
            std::cout << "Verification failed at index " << i << std::endl;
            break;
        }
    }

    // 取消映射SVM内存
    err = clEnqueueSVMUnmap(queue(), c, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Error unmapping SVM memory c: " << err << std::endl;
        return -1;
    }

    // 释放SVM内存
    clSVMFree(context(), a);
    clSVMFree(context(), b);
    clSVMFree(context(), c);

    std::cout << "Computation completed successfully." << std::endl;
    return 0;
}
