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
        // printf("gid: %d, a,b: %f,%f\n", gid, a[gid],b[gid]);

        if (gid < n) {
            c[gid] = a[gid] + b[gid];
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
    const int n = 64; // 1024;
    std::vector<float> a(n, 1.0f);
    std::vector<float> b(n, 2.0f);
    std::vector<float> c(n, 0.0f);

    try {
        // 获取所有平台（例如，NVIDIA、Intel、AMD等）
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            throw std::runtime_error("No OpenCL platforms found");
        }

        // 选择第一个平台
        cl::Platform platform = platforms.front(); // Arc 750
        // cl::Platform platform = platforms[1]; // iGPU UHD 770

        // 获取平台上的所有设备（例如，CPU、GPU等）
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty()) {
            throw std::runtime_error("No OpenCL devices found");
        }

        // 选择第一个设备
        cl::Device device = devices.front();
        printDeviceInfo(device);

        // 创建上下文和命令队列
        cl::Context context(device);
        cl::CommandQueue queue(context, device);

        // // 读取并编译OpenCL kernel
        cl::Program program(context, kernelSource);
        program.build({device});

        // // 创建缓冲区
        // cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, a.data());
        // cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, b.data());
        // cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeof(float) * n);

        // 分配SVM内存
        const int arraySize = n; // 2*1024*1024; // 64; // 32; //  256; // 1024;
        float* a = (float*)clSVMAlloc(context(), CL_MEM_READ_WRITE, sizeof(float) * arraySize, 0);
        float* b = (float*)clSVMAlloc(context(), CL_MEM_READ_WRITE, sizeof(float) * arraySize, 0);
        float* c = (float*)clSVMAlloc(context(), CL_MEM_READ_WRITE, sizeof(float) * arraySize, 0);

        // 映射SVM内存:写 no need for iGPU, needed for dGPU
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
            b[i] = i * 10;
            c[i] = -1;
            // std::cout << "i: " << i << " a,b,c: " << a[i] << "," << b[i] << "," << c[i] << std::endl;
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

        // // 设置kernel参数
        // cl::Kernel kernel(program, "vectorAdd");
        // kernel.setArg(0, bufferA);
        // kernel.setArg(1, bufferB);
        // kernel.setArg(2, bufferC);
        // kernel.setArg(3, n);

        // 设置kernel参数
        cl::Kernel kernel(program, "vectorAdd");
        kernel.setArg(0, a);
        kernel.setArg(1, b);
        kernel.setArg(2, c);
        kernel.setArg(3, arraySize);

        // 执行kernel
        // cl::NDRange global(n);
        cl::NDRange global(arraySize);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);

        // 映射SVM内存:读
        err = clEnqueueSVMMap(queue(), CL_TRUE, CL_MAP_READ, c, arraySize, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Error mapping SVM memory c: " << err << std::endl;
            return -1;
        }

        // 读取结果
        //queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(float) * n, c.data());

        // 验证结果
        for (int i = 0; i < arraySize; i++) {
            // std::cout << "i: " << i << ", Expected: " << i+i*10 << ", got: " << c[i] << std::endl;
            if (c[i] != i + i * 10) {
                std::cout << "Verification failed at index " << i << std::endl;
                //break;
            }
        }

        // // 打印结果
        // for (int i = 0; i < 10; ++i) {
        //     std::cout << "c[" << i << "] = " << c[i] << std::endl;
        // }

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

    //} catch (const cl::Error& err) {
    //    std::cerr << "OpenCL error: " << err.what() << " (" << err.err() << ")" << std::endl;
    //    return 1;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
