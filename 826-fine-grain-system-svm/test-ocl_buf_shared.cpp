#include <iostream>
#include <string>
#include <vector>
#include <CL/cl.h>
#include <chrono>

cl_device_id choose_ocl_device(size_t id)
{
    cl_uint platformCount;
    clGetPlatformIDs(0, nullptr, &platformCount);
    std::vector<cl_platform_id> platforms(platformCount);
    clGetPlatformIDs(platformCount, platforms.data(), nullptr);
    std::vector<cl_device_id> devices_vec;
    for (auto platform : platforms)
    {
        cl_uint deviceCount;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount);
        std::vector<cl_device_id> devices(deviceCount);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, deviceCount, devices.data(), nullptr);
        for (auto device : devices)
        {
            char deviceName[128];
            clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
            // std::cout << "Device: " << deviceName; // << std::endl;
            cl_device_type device_type;
            clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(device_type), &device_type, nullptr);
            // std::cout << ", device_type: " << device_type << std::endl;
        }

        for (auto device : devices)
        {
            cl_device_type device_type;
            clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(device_type), &device_type, nullptr);
            if (device_type == CL_DEVICE_TYPE_GPU)
            {
                char deviceName[128];
                if (devices_vec.size() == id)
                {
                    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
                    // std::cout << "Chosen device: " << deviceName << std::endl;
                }
                devices_vec.push_back(device);
            }
        }
    }
    if (devices_vec.size() > id)
        return devices_vec[id];
    return 0;
}

class oclStruct
{
public:
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem clbuf;
    void deinit()
    {
        clReleaseMemObject(clbuf);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
    }
};

void ocl_share_cpu_mem_across_device()
{
    // Create host buffer
    // size_t N = 256; //64; // 32; //1600; // 350; // 16;
    size_t M = 256; //2048; //256; //64; // 32;
    size_t N = 480; //1188; //480; //428; // 297; //160; //60; //107; //40;
    std::vector<float> host_buf(M * N, 1.0f);
    printf("--> host_buf size %ld \n", sizeof(float) * host_buf.size());
    size_t device_num = 2;

    oclStruct ocl_struct[2];
    for (size_t i = 0; i < device_num; i++)
    {
        // Choose device with OpenCL graphics
        ocl_struct[i].device = choose_ocl_device(i);
        ocl_struct[i].context = clCreateContext(nullptr, 1, &ocl_struct[i].device, nullptr, nullptr, nullptr);
        cl_command_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
        ocl_struct[i].queue = clCreateCommandQueueWithProperties(ocl_struct[i].context, ocl_struct[i].device, props, nullptr);

        auto start_buf = std::chrono::high_resolution_clock::now();
        ocl_struct[i].clbuf = clCreateBuffer(ocl_struct[i].context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * M * N, host_buf.data(), nullptr);
        auto end_buf = std::chrono::high_resolution_clock::now();
        auto ts_create_buf = std::chrono::duration_cast<std::chrono::microseconds>(end_buf - start_buf).count();
        printf("test%ld ts_create_buf: %ld \n", i, ts_create_buf);
    }

    const char *kernel_source = R"(
        __kernel void matrix_add(__global float* A, __global float* B) {
            int i = get_global_id(0);
            int j = get_global_id(1);
            int index = i * get_global_size(1) + j;
            A[index] += B[index];
            // printf("index = %d, res = %f\n", index, A[index]);
        }
    )";

    for (size_t i = 0; i < device_num; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        ocl_struct[i].program = clCreateProgramWithSource(ocl_struct[i].context, 1, &kernel_source, nullptr, nullptr);
        auto ret = clBuildProgram(ocl_struct[i].program, 1, &ocl_struct[i].device, nullptr, nullptr, nullptr);
        auto end_program = std::chrono::high_resolution_clock::now();
        // if (ret == CL_BUILD_PROGRAM_FAILURE)
        // {
        //     printf("Could not build Kernel!\n");
        //     // Determine the size of the log
        //     size_t log_size;
        //     printf(" ret: %i\n", clGetProgramBuildInfo(ocl_struct[i].program, ocl_struct[i].device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));

        //     // Allocate memory for the log
        //     char *log = (char *)malloc(log_size);

        //     // Get the log
        //     printf(" ret: %i\n", clGetProgramBuildInfo(ocl_struct[i].program, ocl_struct[i].device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL));

        //     // Print the log
        //     printf(" ret-val: %i\n", ret);
        //     printf("%s\n", log);
        //     free(log);
        // }

        auto start_kernel = std::chrono::high_resolution_clock::now();
        ocl_struct[i].kernel = clCreateKernel(ocl_struct[i].program, "matrix_add", nullptr);
        auto end_kernel = std::chrono::high_resolution_clock::now();

        if (i != 0)
            clEnqueueWriteBuffer(ocl_struct[i].queue, ocl_struct[i].clbuf, CL_TRUE, 0, sizeof(float) * M * N, host_buf.data(), 0, nullptr, nullptr);

        // auto err  = clEnqueueMigrateMemObjects(ocl_struct[i].queue, 1, &ocl_struct[i].clbuf, 0, 0, nullptr, nullptr);
        // ocl_struct[i].clbuf = clCreateBuffer(ocl_struct[i].context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * N * N, host_buf.data(), nullptr);
        auto start_args = std::chrono::high_resolution_clock::now();
        clSetKernelArg(ocl_struct[i].kernel, 0, sizeof(cl_mem), &ocl_struct[i].clbuf);
        clSetKernelArg(ocl_struct[i].kernel, 1, sizeof(cl_mem), &ocl_struct[i].clbuf);
        auto end_args = std::chrono::high_resolution_clock::now();

        size_t global_work_size[2] = {M, N};
        clEnqueueNDRangeKernel(ocl_struct[i].queue, ocl_struct[i].kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
        auto end_enqueue_kernel = std::chrono::high_resolution_clock::now();

        clFlush(ocl_struct[i].queue);
        clFinish(ocl_struct[i].queue);
        auto end_finish = std::chrono::high_resolution_clock::now();
        clEnqueueReadBuffer(ocl_struct[i].queue, ocl_struct[i].clbuf, CL_TRUE, 0, sizeof(float) * M * N, host_buf.data(), 0, nullptr, nullptr);
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

        // // Read the output buffer back to the host
        // std::vector<float> cl_result(N * N);
        // clEnqueueReadBuffer(ocl_struct[i].queue, ocl_struct[i].clbuf, CL_TRUE, 0, sizeof(float) * N * N, cl_result.data(), 0, nullptr, nullptr);

        std::cout << std::endl
                  << "result " << i << ":" << std::endl;
        for (size_t j = 0; j < 8; j++)
        {
            // std::cout << "\tres[" << j << "]=" << cl_result[j] << std::endl;
            std::cout << "\tres[" << M*N - 1 - j << "]=" << host_buf[M*N-1-j] << std::endl;
        }
    }

    for (size_t i = 0; i < device_num; i++)
    {
        ocl_struct[0].deinit();
    }
}

int main(int argc, char **argv)
{
    ocl_share_cpu_mem_across_device();
    return 1;
}
