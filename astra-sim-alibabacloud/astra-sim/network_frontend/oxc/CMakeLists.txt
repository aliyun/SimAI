# CMake requirement
cmake_minimum_required(VERSION 3.15)

# 项目名称和设置
project(SimAI_oxc)

# 查找源文件
file(GLOB SOURCES "*.cc")
file(GLOB HEADERS "*.h")
include_directories("${PROJECT_SOURCE_DIR}/../../../")

# 查找 libcurl
find_package(CURL REQUIRED)

# 设置可执行文件
add_executable(SimAI_oxc ${SOURCES} ${HEADERS})

# 链接库
target_link_libraries(SimAI_oxc AstraSim CURL::libcurl)
