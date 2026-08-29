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

#define POTTS_TEST_JOIN_IMPL(a, b) a##b
#define POTTS_TEST_JOIN(a, b) POTTS_TEST_JOIN_IMPL(a, b)

#define POTTS_TEST_CASE(name)                                                   \
    static void POTTS_TEST_JOIN(potts_test_function_, __LINE__)();              \
    static TestRegistrar POTTS_TEST_JOIN(potts_test_registrar_, __LINE__)(      \
        name, &POTTS_TEST_JOIN(potts_test_function_, __LINE__));                \
    static void POTTS_TEST_JOIN(potts_test_function_, __LINE__)()

#define POTTS_REQUIRE(expression) \
    test_require(static_cast<bool>(expression), #expression, __FILE__, __LINE__)
