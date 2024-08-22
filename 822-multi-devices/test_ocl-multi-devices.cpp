// #include <CL/cl.hpp>
#include <CL/cl2.hpp>

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

const char* kernelSource = R"(
    __kernel void vectorAdd(__global const float* a,
                        __global const float* b,
                        __global float* c,
                        const int n)
    {
        int gid = get_global_id(0);

        if (gid < n) {
            c[gid] = a[gid] + b[gid];
            // printf("vectorAdd gid: %d, a,b,c: %f,%f,%f\n", gid, a[gid],b[gid],c[gid]);
        }
    }

    __kernel void vectorSub(__global const float* a,
                        __global const float* b,
                        __global float* c,
                        const int n)
    {
        int gid = get_global_id(0);

        if (gid < n) {
            c[gid] = b[gid] - a[gid];
            // printf("vectorSub gid: %d, a,b,c: %f,%f,%f\n", gid, a[gid],b[gid],c[gid]);
        }
    }
)";

// 读取OpenCL kernel文件
std::string readKernelFile(const std::string& fileName) {
    std::ifstream file(fileName);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open kernel file");
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

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
    // 向量大小
    const int dev_num = 2;
    const int n = 256 * 1024 * 1024; // 64; // 1024;
    std::vector<float> a(n, 1.0f);
    std::vector<float> b(n, 2.0f);
    std::vector<float> c(n, 0.0f);
    // std::vector<float> c(n * dev_num, -1.0f);

    try {
        // 获取所有平台（例如，NVIDIA、Intel、AMD等）
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            throw std::runtime_error("No OpenCL platforms found");
        }

        // 遍历所有平台，查找 Intel GPU 平台
        cl::Platform target_platform;
        bool found = false;
        for (const auto& platform : platforms) {
            std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
            std::string platformVendor = platform.getInfo<CL_PLATFORM_VENDOR>();

            std::cout << "Platform Name: " << platformName << std::endl;
            std::cout << "Platform Vendor: " << platformVendor << std::endl;

            // 检查平台是否为 Intel
            if (platformVendor.find("Intel") != std::string::npos) {
                target_platform = platform;
                found = true;
                break;
            }
        }

        if (!found) {
            std::cerr << "Intel GPU platform not found." << std::endl;
            return -1;
        }
        std::cout << "Found Intel GPU platform: " << target_platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

        // 获取平台上的所有设备（例如，CPU、GPU等）
        std::vector<cl::Device> devices;
        target_platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty()) {
            throw std::runtime_error("No OpenCL devices found");
        }

        // 选择第一个设备
        // cl::Device device = devices.front();
        cl::Device device0 = devices[0];
        printDeviceInfo(device0);
        cl::Device device1 = devices[1];
        printDeviceInfo(device1);

        // 创建上下文和命令队列
        // cl::Context context(device);
        // cl::CommandQueue queue(context, device);

        cl::Context context(devices);
        cl::CommandQueue queue0(context, device0);
        cl::CommandQueue queue1(context, device1);

        // // 读取并编译OpenCL kernel
        // std::string kernelSource = readKernelFile("vectorAdd.cl");
        // cl::Program::Sources sources(1, std::make_pair(kernelSource.c_str(), kernelSource.length()));
        // cl::Program program(context, sources);
        cl::Program program(context, kernelSource);
        // program.build({device});
        program.build({devices});

        // 创建缓冲区
        cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, a.data());
        cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, b.data());
        cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeof(float) * n);
        // cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeof(float) * n * dev_num);

        // 设置kernel参数
        cl::Kernel kernel0(program, "vectorAdd");
        kernel0.setArg(0, bufferA);
        kernel0.setArg(1, bufferB);
        kernel0.setArg(2, bufferC);
        kernel0.setArg(3, n);

        // 执行kernel
        cl::NDRange global(n);
        queue0.enqueueNDRangeKernel(kernel0, cl::NullRange, global, cl::NullRange);
        // 读取结果
        size_t copy_len = sizeof(float) * n;
        queue0.enqueueReadBuffer(bufferC, CL_TRUE, 0, copy_len, c.data());
        // queue0.finish();
        // 打印结果
        const int N = 10;
        for (int i = 0; i < N; ++i) {
            std::cout << "c[" << n-N+i << "] = " << c[n-N+i] << std::endl;
        }
        std::cout << std::endl;

        // 创建缓冲区
        cl::Buffer bufferA1(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, a.data());
        cl::Buffer bufferB1(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, b.data());
        cl::Buffer bufferC1(context, CL_MEM_WRITE_ONLY, sizeof(float) * n);
        cl::Kernel kernel1(program, "vectorSub");
        kernel1.setArg(0, bufferA1);
        kernel1.setArg(1, bufferB1);
        kernel1.setArg(2, bufferC1);
        kernel1.setArg(3, n);

        queue1.enqueueNDRangeKernel(kernel1, cl::NullRange, global, cl::NullRange);
        // queue1.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(float) * n * dev_num, c.data());
        queue1.enqueueReadBuffer(bufferC1, CL_TRUE, 0, copy_len, c.data());
        // 打印结果
        for (int i = 0; i < N; ++i) {
            std::cout << "c[" << n-N+i << "] = " << c[n-N+i] << std::endl;
        }
        std::cout << std::endl;

        // check in-place kernel: access 2 buffs on different devices
        cl::Kernel kernel2(program, "vectorSub");
        kernel2.setArg(0, bufferC1);
        kernel2.setArg(1, bufferC);
        kernel2.setArg(2, bufferC1);
        kernel2.setArg(3, n);
        queue1.enqueueNDRangeKernel(kernel2, cl::NullRange, global, cl::NullRange);
        // queue1.enqueueReadBuffer(bufferC1, CL_TRUE, 0, copy_len, c.data());

        // queue1 kernel + queue0 read is also ok
        queue0.enqueueReadBuffer(bufferC1, CL_TRUE, 0, copy_len, c.data());
        // 打印结果
        for (int i = 0; i < N; ++i) {
            std::cout << "c[" << n-N+i << "] = " << c[n-N+i] << std::endl;
        }
        std::cout << std::endl;

    //} catch (const cl::Error& err) {
    //    std::cerr << "OpenCL error: " << err.what() << " (" << err.err() << ")" << std::endl;
    //    return 1;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
