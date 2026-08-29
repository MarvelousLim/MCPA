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

template <typename Exception, typename Function>
inline void test_require_throws(Function&& function, const char* expression,
                                const char* file, int line) {
    try {
        function();
    } catch (const Exception&) {
        return;
    } catch (...) {
        std::ostringstream message;
        message << file << ':' << line << ": wrong exception from: " << expression;
        throw std::runtime_error(message.str());
    }
    std::ostringstream message;
    message << file << ':' << line << ": expected exception from: " << expression;
    throw std::runtime_error(message.str());
}

#define ISING_TEST_JOIN_IMPL(a, b) a##b
#define ISING_TEST_JOIN(a, b) ISING_TEST_JOIN_IMPL(a, b)

#define ISING_TEST_CASE(name)                                                   \
    static void ISING_TEST_JOIN(ising_test_function_, __LINE__)();              \
    static TestRegistrar ISING_TEST_JOIN(ising_test_registrar_, __LINE__)(      \
        name, &ISING_TEST_JOIN(ising_test_function_, __LINE__));                \
    static void ISING_TEST_JOIN(ising_test_function_, __LINE__)()

#define ISING_REQUIRE(expression) \
    test_require(static_cast<bool>(expression), #expression, __FILE__, __LINE__)

#define ISING_REQUIRE_THROWS(exception_type, expression)                        \
    test_require_throws<exception_type>([&]() { expression; }, #expression,     \
                                        __FILE__, __LINE__)
