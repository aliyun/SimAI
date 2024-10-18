# CMake requirement
cmake_minimum_required(VERSION 3.15)

# 项目名称和设置
project(SimAI_analytical)

# 查找源文件
file(GLOB SOURCES "*.cc") # 会查找当前目录下的所有 .cpp 文件
file(GLOB HEADERS "*.h")    # 会查找当前目录下的所有 .h 文件
include_directories("${PROJECT_SOURCE_DIR}/../../../")


# 设置可执行文件
add_executable(SimAI_analytical ${SOURCES} ${HEADERS})

# 链接库
target_link_libraries(SimAI_analytical AstraSim) # 替换为实际需要链接的库名
