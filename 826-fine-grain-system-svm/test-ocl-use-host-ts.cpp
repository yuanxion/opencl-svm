// #include <CL/cl.hpp>
#include <CL/cl2.hpp>

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>

const char* kernelSource = R"(
    __kernel void vectorAdd(__global const float* a,
                        __global const float* b,
                        __global float* c,
                        const int n)
    {
        int gid = get_global_id(0);

        if (gid < n) {
            c[gid] = a[gid] + b[gid];
            printf("vectorAdd gid: %d, a,b,c: %f,%f,%f\n", gid, a[gid],b[gid],c[gid]);
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
            printf("vectorSub gid: %d, a,b,c: %f,%f,%f\n", gid, a[gid],b[gid],c[gid]);
        }
    }
)";

const char *kernel_source = R"(
        __kernel void matrix_add(__global float* A, __global float* B) {
            int i = get_global_id(0);
            int j = get_global_id(1);
            int index = i * get_global_size(1) + j;
            A[index] += B[index];
            // printf("[%d,%d] index = %d, res = %f\n", i, j, index, A[index]);
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
        // printDeviceInfo(device0);
        cl::Device device1 = devices[1];
        // printDeviceInfo(device1);


        // 1. 创建缓冲区
        size_t i = 0;
        // Create host buffer
        // size_t N = 256; //64; // 32; // 1600; // 350; // 16;
        size_t M = 256; //2048; //256; // 64; //32;
        size_t N = 480; //1188; // 480; // 428; // 297; // 160; //60; //107; // 40;

        // std::vector<float> host_buf(N * N, 1.0f);
        std::vector<float> host_buf(M * N, 1.0f);
        printf("--> host_buf size %ld \n", sizeof(float) * host_buf.size());

        // 创建上下文和命令队列
        cl::Context context0(device0);
        cl::CommandQueue queue0(context0, device0);
        auto start_buf = std::chrono::high_resolution_clock::now();
        cl::Buffer bufferA(context0, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * M * N, host_buf.data());
        auto end_buf = std::chrono::high_resolution_clock::now();
        auto ts_create_buf = std::chrono::duration_cast<std::chrono::microseconds>(end_buf - start_buf).count();
        printf("test%ld ts_create_buf: %ld \n", i, ts_create_buf);
        i += 1;
        cl::Context context1(device1); // ok
        cl::CommandQueue queue1(context1, device1);
        start_buf = std::chrono::high_resolution_clock::now();
        cl::Buffer bufferA1(context1, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * M * N, host_buf.data());
        end_buf = std::chrono::high_resolution_clock::now();
        ts_create_buf = std::chrono::duration_cast<std::chrono::microseconds>(end_buf - start_buf).count();
        printf("test%ld ts_create_buf: %ld \n", i, ts_create_buf);

        // // 读取并编译OpenCL kernel
        i = 0;
        auto start = std::chrono::high_resolution_clock::now();
        cl::Program program0(context0, kernel_source);
        program0.build({device0});
        auto end_program = std::chrono::high_resolution_clock::now();

        // 设置kernel参数
        auto start_kernel = std::chrono::high_resolution_clock::now();
        cl::Kernel kernel0(program0, "matrix_add");
        auto end_kernel = std::chrono::high_resolution_clock::now();

        auto start_args = std::chrono::high_resolution_clock::now();
        kernel0.setArg(0, bufferA);
        kernel0.setArg(1, bufferA);
        auto end_args = std::chrono::high_resolution_clock::now();

        // 执行kernel
        cl::NDRange global({M, N});
        queue0.enqueueNDRangeKernel(kernel0, cl::NullRange, global, cl::NullRange);
        auto end_enqueue_kernel = std::chrono::high_resolution_clock::now();
        queue0.finish();
        auto end_finish = std::chrono::high_resolution_clock::now();
        // 读取结果
        size_t copy_size = sizeof(float) * M * N;
        queue0.enqueueReadBuffer(bufferA, CL_TRUE, 0, copy_size, host_buf.data());
        auto end_read = std::chrono::high_resolution_clock::now();

        auto ts_program = std::chrono::duration_cast<std::chrono::microseconds>(end_program - start).count();
        auto ts_create_kernel = std::chrono::duration_cast<std::chrono::microseconds>(end_kernel - end_program).count();
        auto ts_write_buf = std::chrono::duration_cast<std::chrono::microseconds>(start_args - end_kernel).count();
        auto ts_args = std::chrono::duration_cast<std::chrono::microseconds>(end_args - start_args).count();
        auto ts_enqueue = std::chrono::duration_cast<std::chrono::microseconds>(end_enqueue_kernel - end_args).count();
        auto ts_wait = std::chrono::duration_cast<std::chrono::microseconds>(end_finish - end_enqueue_kernel).count();
        auto ts_read_buf = std::chrono::duration_cast<std::chrono::microseconds>(end_read - end_finish).count();
        auto ts_sum = std::chrono::duration_cast<std::chrono::microseconds>(end_read - start).count();
        printf("test%ld ts_program: %ld \n", i, ts_program);
        printf("test%ld ts_create_kernel: %ld \n", i, ts_create_kernel);
        printf("test%ld ts_write_buf: %ld \n", i, ts_write_buf);
        printf("test%ld ts_args: %ld \n", i, ts_args);
        printf("test%ld ts_enqueue: %ld \n", i, ts_enqueue);
        printf("test%ld ts_wait: %ld \n", i, ts_wait);
        printf("test%ld ts_read_buf: %ld \n", i, ts_read_buf);
        printf("test%ld ts_sum: %ld \n", i, ts_sum);

        // 打印结果
        const int n = 8;
        for (int i = 0; i < n; ++i) {
            std::cout << "host_buf[" << M*N-1-i << "] = " << host_buf[M*N-1-i] << std::endl;
        }
        std::cout << std::endl;

        // 1. 创建缓冲区
        i += 1;

        start = std::chrono::high_resolution_clock::now();
        // cl::Program program1(context1, kernelSource);
        cl::Program program1(context1, kernel_source);
        program1.build({device1});
        end_program = std::chrono::high_resolution_clock::now();

        cl::Kernel kernel1(program1, "matrix_add"); // ok, same program0
        end_kernel = std::chrono::high_resolution_clock::now();

        // update results from host_buf changed by GPU0
        queue1.enqueueWriteBuffer(bufferA1, CL_TRUE, 0, copy_size, host_buf.data());

        start_args = std::chrono::high_resolution_clock::now();
        kernel1.setArg(0, bufferA1);
        kernel1.setArg(1, bufferA1);
        end_args = std::chrono::high_resolution_clock::now();

        queue1.enqueueNDRangeKernel(kernel1, cl::NullRange, global, cl::NullRange);
        end_enqueue_kernel = std::chrono::high_resolution_clock::now();
        queue1.finish();
        end_finish = std::chrono::high_resolution_clock::now();

        queue1.enqueueReadBuffer(bufferA1, CL_TRUE, 0, copy_size, host_buf.data());
        end_read = std::chrono::high_resolution_clock::now();

        ts_program = std::chrono::duration_cast<std::chrono::microseconds>(end_program - start).count();
        ts_create_kernel = std::chrono::duration_cast<std::chrono::microseconds>(end_kernel - end_program).count();
        ts_write_buf = std::chrono::duration_cast<std::chrono::microseconds>(start_args - end_kernel).count();
        ts_args = std::chrono::duration_cast<std::chrono::microseconds>(end_args - start_args).count();
        ts_enqueue = std::chrono::duration_cast<std::chrono::microseconds>(end_enqueue_kernel - end_args).count();
        ts_wait = std::chrono::duration_cast<std::chrono::microseconds>(end_finish - end_enqueue_kernel).count();
        ts_read_buf = std::chrono::duration_cast<std::chrono::microseconds>(end_read - end_finish).count();
        ts_sum = std::chrono::duration_cast<std::chrono::microseconds>(end_read - start).count();
        printf("test%ld ts_program: %ld \n", i, ts_program);
        printf("test%ld ts_create_kernel: %ld \n", i, ts_create_kernel);
        printf("test%ld ts_write_buf: %ld \n", i, ts_write_buf);
        printf("test%ld ts_args: %ld \n", i, ts_args);
        printf("test%ld ts_enqueue: %ld \n", i, ts_enqueue);
        printf("test%ld ts_wait: %ld \n", i, ts_wait);
        printf("test%ld ts_read_buf: %ld \n", i, ts_read_buf);
        printf("test%ld ts_sum: %ld \n", i, ts_sum);

        // 打印结果
        for (int i = 0; i < n; ++i) {
            std::cout << "host_buf[" << M*N-1-i << "] = " << host_buf[M*N-1-i] << std::endl;
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
