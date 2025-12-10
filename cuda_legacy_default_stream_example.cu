#include <cstdio>
#include <cuda_runtime.h>

void default_wait_explict()
{

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    // 假设GPU支持并发内核执行

    // 在非默认流 stream1 上启动一个长时间运行的内核
    kernelLong<<<grid1, block1, 0, stream1>>>();

    // 紧接着，在Legacy Default Stream（流0）上启动一个内核
    kernelShort<<<grid2, block2>>>(); // 未指定流，即使用默认流

    // 你以为的执行时间线 (期望并发):
    // |----- kernelLong (stream1) -----|
    //           |-- kernelShort (default) --|
    //
    // 实际的执行时间线 (因Legacy Default Stream):
    // |----- kernelLong (stream1) -----|
    //                                   |-- kernelShort (default) --|
    //
    // kernelShort 会一直等待 kernelLong 完成才开始，尽管它们在逻辑上无关。
}

void explit_wait_default()
{
    cudaStream_t streamA;
    cudaStreamCreate(&streamA);

    // 情景1: 默认流等待非默认流
    kernel1<<<..., streamA>>>();             // 在流A中
    cudaMemcpy(..., cudaMemcpyHostToDevice); // 在默认流中。本次拷贝会**等待** kernel1 完成后才开始。

    // 情景2: 默认流阻塞非默认流
    kernel2<<<...>>>();          // 在默认流中启动一个耗时内核
    kernel3<<<..., streamA>>>(); // 在流A中。kernel3 会被**阻塞**，直到默认流中的 kernel2 完成后才开始。
}