cmake_minimum_required(VERSION 3.16.3)
project(resnet)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/usr/local/libtorch;/usr/local/opencv3")
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

aux_source_directory(./src SRCS)
ADD_EXECUTABLE(${PROJECT_NAME} ${SRCS})

target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)