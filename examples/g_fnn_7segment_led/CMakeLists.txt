cmake_minimum_required(VERSION 3.10)

project(g_fnn_7segment_led VERSION 1.0)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# set(CMAKE_BUILD_TYPE Debug)

# set(CMAKE_BUILD_TYPE Release)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../build)

add_compile_options(-Wall -Wextra -pedantic)

include_directories(
    ../
    ../../src
)

add_executable(
    "g_fnn_7segment_led"
    "../data_reader.c"
    "../data_writer.c"
    "../../src/g_page.c"
    "../../src/g_neuron.c"
    "../../src/g_layer.c"
    "../../src/g_network.c"
    "../../src/g_random.c"
    "fnn_layout.c"
    "main.c"
)

target_link_libraries("g_fnn_7segment_led" m)

# target_compile_definitions(g_fnn_7segment_led PUBLIC MY_MACRO=1)
