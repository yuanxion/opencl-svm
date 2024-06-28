#include <mpi.h>
//#include <CL/cl.h>
#include <CL/cl2.hpp>
#include <iostream>
#include <vector>

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        std::cerr << "OpenCL error: " << err << " at line " << __LINE__ << std::endl; \
        exit(1); \
    }

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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::cout << "rank: " << rank << ", size: " << size << std::endl;

    if (size != 2) {
        std::cerr << "This program requires exactly 2 processes" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    try {
        // 获取所有平台（例如，NVIDIA、Intel、AMD等）
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            throw std::runtime_error("No OpenCL platforms found");
        }

        // 选择第一个平台
        //cl::Platform platform = platforms.front(); // Arc 750
        //cl::Platform platform = platforms[1]; // iGPU UHD 770
        cl::Platform platform = platforms[rank]; // iGPU UHD 770

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


        // 分配GPU内存
        const int dataSize = 1024;
        // cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, dataSize * sizeof(int), NULL, &err);

        if (rank == 0) {
            // 进程0初始化数据
            std::vector<int> hostData(dataSize);
            for (int i = 0; i < dataSize; ++i) {
                hostData[i] = i;
            }
            // err = clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, dataSize * sizeof(int), hostData.data(), 0, NULL, NULL);

            cl::Buffer buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * dataSize, hostData.data());

            // 发送数据到进程1
            MPI_Send(hostData.data(), dataSize, MPI_INT, 1, 0, MPI_COMM_WORLD);
        } else {
            // 进程1接收数据
            std::vector<int> hostData(dataSize);
            MPI_Recv(hostData.data(), dataSize, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // 将接收到的数据写入GPU内存
            cl::Buffer buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * dataSize, hostData.data());
            // 读取结果
            queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(float) * dataSize, hostData.data());

            bool valid = true;
            for (int i = 0; i < dataSize; ++i) {
                if (hostData[i] != i) {
                    valid = false;
                    break;
                } else if (i < 10) {
                    std::cout << "Received hostData[" << i << "]: " << hostData[i] << std::endl;
                }
            }
            std::cout << "Data transfer " << (valid ? "successful" : "failed") << std::endl;
        }


    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    MPI_Finalize();
    return 0;
}

