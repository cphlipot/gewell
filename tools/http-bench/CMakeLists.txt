cmake_minimum_required(VERSION 3.25)
project(gewell_http_bench VERSION 0.1.0 LANGUAGES CXX)

# This project deliberately configures on its own, without the inference build.
find_package(CURL 7.85 REQUIRED)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)
add_executable(gewell-http-bench main.cc)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS main.cc)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/main.cc" GEWELL_BENCH_SOURCE_SHA256)
target_compile_features(gewell-http-bench PRIVATE cxx_std_17)
target_include_directories(gewell-http-bench SYSTEM PRIVATE ../../vendor/nlohmann)
target_link_libraries(gewell-http-bench PRIVATE CURL::libcurl OpenSSL::Crypto)
target_compile_definitions(gewell-http-bench PRIVATE
  GEWELL_BENCH_VERSION="${PROJECT_VERSION}"
  GEWELL_BENCH_SOURCE_SHA256="${GEWELL_BENCH_SOURCE_SHA256}"
  GEWELL_BENCH_BUILD="${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}/${CMAKE_BUILD_TYPE}")
target_compile_options(gewell-http-bench PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>)
install(TARGETS gewell-http-bench RUNTIME DESTINATION bin)

option(BUILD_TESTING "Build the HTTP fixture tests (requires Python)" OFF)
include(CTest)
if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(NAME http_benchmark_protocol_and_load
    COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/test_http_bench.py"
      "$<TARGET_FILE:gewell-http-bench>")
  set_tests_properties(http_benchmark_protocol_and_load PROPERTIES TIMEOUT 120)
endif()
