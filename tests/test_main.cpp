#include "test_harness.hpp"

#include <cstring>
#include <exception>
#include <iostream>

std::vector<TestCase>& test_registry() {
    static std::vector<TestCase> tests;
    return tests;
}

TestRegistrar::TestRegistrar(const char* name, void (*function)()) {
    test_registry().push_back(TestCase{name, function});
}

int main(int argc, char* argv[]) {
    const char* selected = nullptr;
    bool list_only = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--list") == 0) {
            list_only = true;
        } else if (std::strncmp(argv[i], "--test=", 7) == 0) {
            selected = argv[i] + 7;
        } else {
            std::cerr << "Usage: " << argv[0] << " [--list|--test=NAME]\n";
            return 2;
        }
    }

    if (list_only) {
        for (const TestCase& test : test_registry()) std::cout << test.name << '\n';
        return 0;
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
        std::cerr << "No test named: " << (selected ? selected : "<none>") << '\n';
        return 2;
    }
    return 0;
}
