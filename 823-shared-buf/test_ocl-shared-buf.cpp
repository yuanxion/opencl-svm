// #include <CL/cl.hpp>
#include <CL/cl2.hpp>

#include <iostream>
#include <vector>

int main() {
    // 获取所有平台
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    // 选择第一个平台
    cl::Platform platform = platforms.front();

    // 获取平台上的所有设备
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

    // 假设我们有至少两个设备
    if (devices.size() < 2) {
        std::cerr << "需要至少两个GPU设备" << std::endl;
        return 1;
    }

    // 创建两个上下文，每个上下文包含一个设备
    cl::Context context1(devices[0]);
    cl::Context context2(devices[1]);

    // 创建命令队列
    cl::CommandQueue queue1(context1, devices[0]);
    cl::CommandQueue queue2(context2, devices[1]);

    // 创建缓冲区
    size_t dataSize = sizeof(float) * 10;
    cl::Buffer buffer(context1, CL_MEM_READ_WRITE, dataSize);

    // 初始化缓冲区数据
    std::vector<float> hostData(10, -1.0f);
    queue1.enqueueWriteBuffer(buffer, CL_TRUE, 0, dataSize, hostData.data());
    // 读取迁移后的数据
    std::vector<float> resultData(10);
    queue1.enqueueReadBuffer(buffer, CL_TRUE, 0, dataSize, resultData.data());
    // 打印结果
    std::cout << "Migrated Data: ";
    for (size_t i = 0; i < resultData.size(); ++i) {
        std::cout << resultData[i] << " ";
    }
    std::cout << std::endl;

    // 将缓冲区迁移到第二个设备 # failed
    cl_mem mem_objects[] = { buffer() };
    clEnqueueMigrateMemObjects(queue2(), 1, mem_objects, 0, 0, nullptr, nullptr);
    queue2.enqueueReadBuffer(buffer, CL_TRUE, 0, dataSize, resultData.data());
    // 打印结果
    std::cout << "Migrated Data: ";
    for (size_t i = 0; i < resultData.size(); ++i) {
        std::cout << resultData[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

