// ----- example.cu -----
#include <stdio.h>
#include <iostream>
using namespace std;
__global__ void kernel() {
    printf("Hello from kernel\n");
}

void kernel_launcher() {
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}

int main() {
    cout << "kernel start" << endl;
    kernel_launcher();

    cout << "kernel end" << endl;
    return 0;
}
