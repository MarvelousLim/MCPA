#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <cuda_runtime.h>
#include <iostream>

int main(int argc, char** argv) {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
        std::cout << "SKIP: CUDA tests need an accessible NVIDIA GPU";
        if (status != cudaSuccess) {
            std::cout << " (" << cudaGetErrorString(status) << ")";
        }
        std::cout << '\n';
        return 77;
    }

    doctest::Context context(argc, argv);
    return context.run();
}
