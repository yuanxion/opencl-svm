// #include <CL/cl.h>
// #include <CL/cl_ext.h>
#include <CL/cl2.hpp>
#include "common.h"

#include <iostream>
#include <vector>

const char* kernelSource = R"(
__kernel void simple_add(__global const float* A, __global const float* B, __global float* C, int size) {
    int id = get_global_id(0);
    if (id < size) {
        C[id] = A[id] + B[id];
        printf("[simple_add: %d] %f + %f = %f\n", id, A[id], B[id], C[id]);
    }
}
)";

int main() {

    // 获取所有可用的 OpenCL 平台
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    // 检查是否找到任何平台
    if (platforms.empty()) {
        std::cerr << "No OpenCL platforms found." << std::endl;
        return -1;
    }

    // 获取第一个平台上的所有 GPU 设备
    std::vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

    // 检查是否找到至少两个设备
    if (devices.size() < 2) {
        std::cerr << "Less than two GPU devices found on the platform." << std::endl;
        return -1;
    }

    // 选择前两个 GPU 设备
    cl::Device device1 = devices[0];
    cl::Device device2 = devices[1];
    std::cout << "Using devices: " << device1.getInfo<CL_DEVICE_NAME>() << " and " << device2.getInfo<CL_DEVICE_NAME>() << std::endl;

    // check USM support
    std::string extensions = device1.getInfo<CL_DEVICE_EXTENSIONS>();
    if (extensions.find("cl_intel_unified_shared_memory") == std::string::npos) {
        std::cerr << "Device does not support Unified Shared Memory (USM)." << std::endl;
        return -1;
    }

    // 创建包含两个 GPU 设备的 OpenCL 上下文
    cl::Context context1(device1);
    // cl::Context context2(device2);
    cl::Context context2(device1);

    // 创建命令队列，分别关联到两个 GPU 设备
    cl::CommandQueue queue1(context1, device1);
    // cl::CommandQueue queue2(context2, device2);
    cl::CommandQueue queue2(context2, device1);

    // 分配 USM 设备内存
    size_t n = 16; // 1024; // 假设缓冲区大小为 1024 个 float
    cl_int err;
    // usm_device
    // float* usm_device_A = (float*)clDeviceMemAllocINTEL(context(), device1(), nullptr, sizeof(float) * n, 0, &err);
    // usm_shared
    // float* usm_device_A = (float*)clSharedMemAllocINTEL(context(), device2(), nullptr, sizeof(float) * n, 0, &err);
    // usm_host
    float* usm_device_A = (float*)clHostMemAllocINTEL(context1(), nullptr, sizeof(float) * n, 0, &err);
    printf("[check] clDeviceMemAllocINTEL: %p \n", usm_device_A);
    // float* usm_device_B = (float*)clDeviceMemAllocINTEL(context(), device1(), nullptr, sizeof(float) * n, 0, &err);
    // float* usm_device_B = (float*)clSharedMemAllocINTEL(context(), device2(), nullptr, sizeof(float) * n, 0, &err);
    float* usm_device_B = (float*)clHostMemAllocINTEL(context1(), nullptr, sizeof(float) * n, 0, &err);
    printf("[check] clSharedMemAllocINTEL: %p \n", usm_device_B);
    // float* usm_device_C = (float*)clDeviceMemAllocINTEL(context(), device1(), nullptr, sizeof(float) * n, 0, &err);
    // float* usm_device_C = (float*)clSharedMemAllocINTEL(context(), device1(), nullptr, sizeof(float) * n, 0, &err);
    // cl_mem usm_device_C = (cl_mem)clSharedMemAllocINTEL(context(), device1(), nullptr, sizeof(cl_mem) * n, 0, &err);
    float* usm_device_C = (float*)clHostMemAllocINTEL(context1(), nullptr, sizeof(float) * n, 0, &err);
    printf("[check] clHostMemAllocINTEL: %p \n", usm_device_C);

    if (err != CL_SUCCESS) {
        std::cerr << "Failed to allocate USM device memory." << std::endl;
        return -1;
    }

    // 检查 USM 设备内存是否分配成功
    if (!usm_device_A || !usm_device_B || !usm_device_C) {
        std::cerr << "USM device memory allocation returned nullptr." << std::endl;
        return -1;
    }

    std::cout << "USM device memory allocated successfully." << std::endl;

    // assign values
    std::vector<float> bufA(n, 1);
    std::vector<float> bufB(n, 2);
    std::vector<float> bufC(n, -1);
    clEnqueueMemcpyINTEL(queue1(), CL_TRUE, usm_device_A, bufA.data(), sizeof(float) * n, 0, nullptr, nullptr);
    std::cout << "clEnqueueMemcpyINTEL usm_device_A done" << std::endl;
    clEnqueueMemcpyINTEL(queue1(), CL_TRUE, usm_device_B, bufB.data(), sizeof(float) * n, 0, nullptr, nullptr);
    // clEnqueueMemcpyINTEL(queue2(), CL_TRUE, usm_device_B, bufB.data(), sizeof(float) * n, 0, nullptr, nullptr);
    std::cout << "clEnqueueMemcpyINTEL usm_device_B done" << std::endl;

    // 创建并编译 OpenCL 程序
    cl::Program program1(context1, kernelSource);
    cl::Program program2(context2, kernelSource);
    program1.build(device1);
    // program2.build(device2);
    program2.build(device1);
    std::cout << "program build done" << std::endl;

    // 创建内核
    cl::Kernel kernel1(program1, "simple_add");
    cl::Kernel kernel2(program2, "simple_add");

    // 设置内核参数
    kernel1.setArg(0, usm_device_A);
    kernel1.setArg(1, usm_device_B);
    kernel1.setArg(2, usm_device_C);
    kernel1.setArg(3, n);

    // 定义全局工作尺寸
    cl::NDRange global(n);

    // 将内核执行命令排入第一个命令队列
    queue1.enqueueNDRangeKernel(kernel1, cl::NullRange, global, cl::NullRange);

    // 使用 cl::CommandQueue::finish 进行同步等待
    queue1.finish();
    std::cout << "Kernel execution completed on device1." << std::endl;
    // print result
    const int N = 16;
    for (int i = 0; i < N; ++i) {
        std::cout << "[" << i << "] " << usm_device_C[i] << std::endl;
    }

    // 尝试在第二个命令队列上进行操作
    // 注意：这可能会失败，具体取决于平台和设备的支持情况
    try {
        // 设置内核参数
        printf("Test cross context usm_host access \n");
        // failed to use usm_host bufs from context1
        // usm_device_A = (float*)clHostMemAllocINTEL(context2(), nullptr, sizeof(float) * n, 0, &err);
        // usm_device_B = (float*)clHostMemAllocINTEL(context2(), nullptr, sizeof(float) * n, 0, &err);
        // usm_device_C = (float*)clHostMemAllocINTEL(context2(), nullptr, sizeof(float) * n, 0, &err);
        clEnqueueMemcpyINTEL(queue2(), CL_TRUE, usm_device_A, bufA.data(), sizeof(float) * n, 0, nullptr, nullptr);
        clEnqueueMemcpyINTEL(queue2(), CL_TRUE, usm_device_B, bufB.data(), sizeof(float) * n, 0, nullptr, nullptr);

        kernel2.setArg(0, usm_device_A);
        kernel2.setArg(1, usm_device_B);
        kernel2.setArg(2, usm_device_C);
        kernel2.setArg(3, n);
        queue2.enqueueNDRangeKernel(kernel2, cl::NullRange, global, cl::NullRange);

        // 使用 cl::CommandQueue::finish 进行同步等待
        queue2.finish();
        std::cout << "Kernel execution completed on device1." << std::endl;
        // print result
        const int N = 16;
        for (int i = 0; i < N; ++i) {
            std::cout << "[" << i << "] " << usm_device_C[i] << std::endl;
        }
    // } catch (cl::Error& e) {
    //     std::cerr << "Error during cross-device operation: " << e.what() << " (" << e.err() << ")" << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    // 释放 USM 设备内存
    clMemFreeINTEL(context1(), usm_device_A);
    clMemFreeINTEL(context1(), usm_device_B);
    clMemFreeINTEL(context1(), usm_device_C);

    printf("Done \n");
    return 0;
}
