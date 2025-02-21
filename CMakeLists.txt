
cmake_minimum_required(VERSION 3.10)

#set(CMAKE_C_COMPILER clang)
#set(CMAKE_CXX_COMPILER clang++)

project(g_fnn VERSION 1.0)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_compile_options(-Wall -Wextra -pedantic)

add_executable(g_fnn
    g_neuron.c
    g_layer.c
    g_network.c
    main.c
)

target_link_libraries(g_fnn m)

# target_compile_definitions(g_fnn PUBLIC MY_MACRO=1)

# set(CMAKE_BUILD_TYPE Release)

