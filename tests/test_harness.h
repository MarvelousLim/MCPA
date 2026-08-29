#pragma once

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct TestCase {
    const char* name;
    void (*function)();
};

std::vector<TestCase>& test_registry();

struct TestRegistrar {
    TestRegistrar(const char* name, void (*function)());
};

inline void test_require(bool condition, const char* expression,
                         const char* file, int line) {
    if (condition) return;
    std::ostringstream message;
    message << file << ':' << line << ": requirement failed: " << expression;
    throw std::runtime_error(message.str());
}

#define BC_TEST_JOIN_IMPL(a, b) a##b
#define BC_TEST_JOIN(a, b) BC_TEST_JOIN_IMPL(a, b)

#define BC_TEST_CASE(name)                                                        \
    static void BC_TEST_JOIN(bc_test_function_, __LINE__)();                      \
    static TestRegistrar BC_TEST_JOIN(bc_test_registrar_, __LINE__)(              \
        name, &BC_TEST_JOIN(bc_test_function_, __LINE__));                        \
    static void BC_TEST_JOIN(bc_test_function_, __LINE__)()

#define BC_REQUIRE(expression) \
    test_require(static_cast<bool>(expression), #expression, __FILE__, __LINE__)
