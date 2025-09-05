cmake_minimum_required(VERSION 3.14)
project(DotPeenReader)

set(CMAKE_CXX_STANDARD 17)

# Find OpenCV with the components we need
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# Find ZXing from your install directory
find_package(ZXing CONFIG REQUIRED)

# Our app
add_executable(dotpeen_reader main.cpp)

# Include OpenCV headers
target_include_directories(dotpeen_reader PRIVATE ${OpenCV_INCLUDE_DIRS})

# Link OpenCV and ZXing
target_link_libraries(dotpeen_reader PRIVATE ${OpenCV_LIBS} ZXing::ZXing)
