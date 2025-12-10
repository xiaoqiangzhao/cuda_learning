#include <iostream>

//example for __syncthread()
__global__ void reduceSum(int* input, int* output) {
    __shared__ int sdata[256];
    int tid = threadIdx.x;
    
    // 1. 每个线程加载数据到共享内存
    sdata[tid] = input[blockIdx.x * blockDim.x + tid];
    __syncthreads(); // 等待所有线程完成加载

    // 2. 在共享内存上进行树状规约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads(); // 等待每轮规约完成，再进入下一轮
    }

    // 3. 线程0将结果写回全局内存
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}


//example for __threadfence()
//to guarantee write--fence--read
__device__ int flag = 0;
__device__ float data;

__global__ void producer() {
    // 生产数据
    data = 3.14f;
    // 确保数据写入对消费者可见后，再更新标志位
    __threadfence(); // 关键：保证 data 的写入先于 flag 的写入被其他线程看到
    flag = 1;
}

__global__ void consumer(float* result) {
    while (atomicAdd(&flag, 0) == 0) {} // 轮询标志位
    __threadfence_acquire(); // 确保读取到 flag==1 后，一定能看到之前生产者写入的 data
    *result = data;
}


//Example for cudaStreamWaitEvent
int main()
{
    cudaStream_t stream1, stream2;
    cudaEvent_t event;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaEventCreate(&event);

    // 流1：从主机拷贝数据A到设备
    cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream1);
    cudaEventRecord(event, stream1); // 记录事件，标记拷贝A完成

    // 流2：等待流1的数据拷贝完成，然后启动内核进行计算B
    cudaStreamWaitEvent(stream2, event, 0); // 关键：流2等待事件
    kernelB<<<..., stream2>>>(d_B, ...);

    // 同时，流1在拷贝完A后，可以继续做其他工作，如计算A
    kernelA<<<..., stream1>>>(d_A, ...);

    // 主机等待所有流完成
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);



//exmaple for cudaEventElapsedTime
    //used for kernel elapsed time record with start and end event
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t start;
    cudaEvent_t stop;

    // create the events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // record the start event
    cudaEventRecord(start, stream);

    // launch the kernel
    kernel<<<grid, block, 0, stream>>>(...);

    // record the stop event
    cudaEventRecord(stop, stream);

    // wait for the stream to complete
    // both events will have been triggered
    cudaStreamSynchronize(stream);

    // get the timing
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Kernel execution time: " << elapsedTime << " ms" << std::endl;

    // clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);

    return 0;
}

//example for cudaStreamQuery()
void use_stream_query()
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // Have a peek at the stream
    // returns cudaSuccess if the stream is empty
    // returns cudaErrorNotReady if the stream is not empty
    cudaError_t status = cudaStreamQuery(stream);

    switch (status)
    {
    case cudaSuccess:
        // The stream is empty
        std::cout << "The stream is empty" << std::endl;
        break;
    case cudaErrorNotReady:
        // The stream is not empty
        std::cout << "The stream is not empty" << std::endl;
        break;
    default:
        // An error occurred - we should handle this
        break;
    };
}