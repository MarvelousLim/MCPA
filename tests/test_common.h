#pragma once
#ifndef TEST_COMMON_H
#define TEST_COMMON_H

#include <iostream>
#include <windows.h>

// ANSI color codes
#define COLOR_RESET   "\033[0m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_RED     "\033[31m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_CYAN    "\033[36m"
#define COLOR_BOLD    "\033[1m"

// Global test counters (defined in tests.cpp, declared here for other files)
extern int g_test_passing;
extern int g_test_failed;

// Global mock parameters
struct Params;  // forward declaration (full definition in baxterwu_lib.h)
extern struct Params mock_params;
extern void setup_mock_params(int L, int R, int N, bool heat);

// Custom assertion macro (available to all test files)
#define ASSERT_TRUE(condition, message) \
    if (!(condition)) { \
        std::cout << COLOR_RED "[FAIL] " COLOR_RESET << __func__ << ": " << message << "\n"; \
        g_test_failed++; \
    } else { \
        std::cout << COLOR_GREEN "[PASS] " COLOR_RESET << __func__ << ": " << message << "\n"; \
        g_test_passing++; \
    } \
    std::cout.flush();

#endif