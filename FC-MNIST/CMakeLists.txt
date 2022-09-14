cmake_minimum_required(VERSION 3.16.3)
project(FCnet)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/usr/local/libtorch")
find_package(Torch REQUIRED)

aux_source_directory(./src SRCS)
add_executable(${PROJECT_NAME} ${SRCS})

target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)