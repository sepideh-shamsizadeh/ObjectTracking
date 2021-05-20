cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 14)

project(object_tracking)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(object_tracking main.cpp)

# linking
target_link_libraries(object_tracking ${OpenCV_LIBS})