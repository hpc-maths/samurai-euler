cmake_minimum_required(VERSION 3.16)
project(euler_tutorial)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)
set(SAMURAI_WITH_MPI ON)
find_package(samurai REQUIRED)

# Extract Samurai version info if it exists, and inject it into the build process
if(NOT DEFINED samurai_VERSION)
    set(samurai_VERSION "no version info")
endif()
add_library(common_defs INTERFACE)
target_compile_definitions(common_defs INTERFACE
    SAMURAI_VERSION="${samurai_VERSION}"
)

# Define executables
add_executable(euler_1d main_1d.cpp)
target_link_libraries(euler_1d PRIVATE samurai::samurai common_defs)

add_executable(euler_user_pred_1d main_user_pred_1d.cpp)
target_link_libraries(euler_user_pred_1d PRIVATE samurai::samurai common_defs)

add_executable(euler_2d main_2d.cpp)
target_link_libraries(euler_2d PRIVATE samurai::samurai common_defs)

add_executable(euler_3d main_3d.cpp)
target_link_libraries(euler_3d PRIVATE samurai::samurai common_defs)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
# The suite drives the built binaries as subprocesses, so it needs to know where
# they are. `ctest` runs the fast tests; `ctest -L slow` adds the validation
# runs, which measure convergence orders and take longer.
option(BUILD_TESTING "Build the tests" ON)

if(BUILD_TESTING)
  enable_testing()
  find_package(Python3 COMPONENTS Interpreter REQUIRED)

  set(EULER_PYTEST ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/tests
                   -c ${CMAKE_CURRENT_SOURCE_DIR}/tests/pytest.ini)

  add_test(NAME tests COMMAND ${EULER_PYTEST})
  add_test(NAME tests-slow COMMAND ${EULER_PYTEST} -m slow)

  set_tests_properties(tests tests-slow PROPERTIES
                       ENVIRONMENT "EULER_BUILD_DIR=${CMAKE_BINARY_DIR}")
  set_tests_properties(tests-slow PROPERTIES LABELS slow)
endif()
