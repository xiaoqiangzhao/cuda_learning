#include <cstdio>
#include <cuda_runtime.h>

// 1. 定义回调函数
//    注意：此函数在CPU线程上执行，可安全调用C标准库等主机API。
void CUDART_CB myStreamCallback(cudaStream_t stream, cudaError_t status, void* userData) {
    printf("[Callback] Stream %p finished with status: %s\n",
           (void*)stream, cudaGetErrorString(status));

    // 2. 从 userData 中提取我们传递的上下文信息
    struct CallbackData* data = (struct CallbackData*)userData;
    
    // 3. 安全地进行后处理（例如：将结果拷贝出固定内存）
    //    注意：由于前置操作已保证完成，这里的拷贝是同步的，但不会阻塞主控制流。
    cudaMemcpyAsync(data->hostResult, data->deviceResult,
                    data->size, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream); // 等待这个最后的拷贝完成

    // 4. 处理结果（例如：验证、保存、触发下一个任务）
    printf("[Callback] First result value: %f\n", data->hostResult[0]);
    printf("[Callback] Computation and processing complete.\n");
    
    // 注意：通常在此释放 userData 中动态分配的内存（本例中在主函数释放）。
}

// 用于传递数据的结构体
struct CallbackData {
    float* deviceResult;
    float* hostResult;
    size_t size;
};

__global__ void myKernel(float* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 模拟一些计算
    result[idx] = idx * 0.1f;
}

int main() {
    const size_t N = 1024;
    const size_t bytes = N * sizeof(float);
    
    // 分配设备内存
    float* d_result;
    cudaMalloc(&d_result, bytes);
    
    // 分配固定主机内存（确保异步拷贝高效）
    float* h_result;
    cudaMallocHost(&h_result, bytes);
    
    // 准备回调数据
    CallbackData cbData = {d_result, h_result, bytes};
    
    // 创建非默认流
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // 启动内核
    myKernel<<<N/256, 256, 0, stream>>>(d_result);
    
    // 将回调函数插入流中
    // 内核完成后，驱动会自动调用 myStreamCallback
    cudaStreamAddCallback(stream, myStreamCallback, &cbData, 0);
    
    // 主线程可以继续执行其他不依赖GPU结果的任务
    printf("[Main] Kernel launched, callback registered. Main thread continues...\n");
    // ... 这里可以执行其他CPU工作 ...
    
    // 最终，等待流中所有操作（包括回调）完成
    cudaStreamSynchronize(stream);
    
    // 清理资源
    cudaFree(d_result);
    cudaFreeHost(h_result);
    cudaStreamDestroy(stream);
    
    return 0;
}