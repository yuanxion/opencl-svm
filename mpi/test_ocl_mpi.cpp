#include <mpi.h>
#include <CL/cl.h>
#include <iostream>
#include <vector>

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        std::cerr << "OpenCL error: " << err << " at line " << __LINE__ << std::endl; \
        exit(1); \
    }

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        std::cerr << "This program requires exactly 2 processes" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // OpenCL 初始化
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    // 获取Intel平台
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err);

    // 获取GPU设备
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(err);

    // 创建上下文和命令队列
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err);
    queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_ERROR(err);

    // 分配GPU内存
    const int dataSize = 1024;
    cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, dataSize * sizeof(int), NULL, &err);
    CHECK_ERROR(err);

    if (rank == 0) {
        // 进程0初始化数据
        std::vector<int> hostData(dataSize);
        for (int i = 0; i < dataSize; ++i) {
            hostData[i] = i;
        }
        err = clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, dataSize * sizeof(int), hostData.data(), 0, NULL, NULL);
        CHECK_ERROR(err);

        // 发送数据到进程1
        MPI_Send(hostData.data(), dataSize, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else {
        // 进程1接收数据
        std::vector<int> hostData(dataSize);
        MPI_Recv(hostData.data(), dataSize, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // 将接收到的数据写入GPU内存
        err = clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, dataSize * sizeof(int), hostData.data(), 0, NULL, NULL);
        CHECK_ERROR(err);

        // 验证接收到的数据
        err = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, dataSize * sizeof(int), hostData.data(), 0, NULL, NULL);
        CHECK_ERROR(err);

        bool valid = true;
        for (int i = 0; i < dataSize; ++i) {
            if (hostData[i] != i) {
                valid = false;
                break;
            }
        }
        std::cout << "Data transfer " << (valid ? "successful" : "failed") << std::endl;
    }

    // 清理资源
    clReleaseMemObject(buffer);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    MPI_Finalize();
    return 0;
}

