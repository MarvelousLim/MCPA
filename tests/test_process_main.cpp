#include "test_harness.hpp"

#include <cuda_runtime.h>

#include <cstring>
#include <exception>
#include <iostream>
#include <string>

namespace {
std::string process_binary;
}

const std::string& ising_process_binary() {
    return process_binary;
}

std::vector<TestCase>& test_registry() {
    static std::vector<TestCase> tests;
    return tests;
}

TestRegistrar::TestRegistrar(const char* name, void (*function)()) {
    test_registry().push_back(TestCase{name, function});
}

int main(int argc, char* argv[]) {
    const char* selected = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--binary=", 9) == 0)
            process_binary = argv[i] + 9;
        else if (std::strncmp(argv[i], "--test=", 7) == 0)
            selected = argv[i] + 7;
        else {
            std::cerr << "Usage: " << argv[0]
                      << " --binary=PATH [--test=NAME]\n";
            return 2;
        }
    }
    if (process_binary.empty()) {
        std::cerr << "Missing --binary=PATH\n";
        return 2;
    }
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
        std::cout << "SKIP: process tests need an accessible NVIDIA GPU\n";
        return 77;
    }

    int executed = 0;
    for (const TestCase& test : test_registry()) {
        if (selected && std::strcmp(selected, test.name) != 0) continue;
        ++executed;
        try {
            test.function();
            std::cout << "PASS: " << test.name << '\n';
        } catch (const std::exception& error) {
            std::cerr << "FAIL: " << test.name << "\n  " << error.what() << '\n';
            return 1;
        }
    }
    if (executed == 0) {
        std::cerr << "No matching process test\n";
        return 2;
    }
    return 0;
}
