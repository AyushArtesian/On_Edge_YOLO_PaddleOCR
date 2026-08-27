cmake_minimum_required(VERSION 3.22.1)

project("yolo_native")

# ---------------------------------------------------------
# NCNN
# Automatically selects the correct architecture:
#
# x86_64     -> Android emulator
# arm64-v8a  -> Physical Android phone
# ---------------------------------------------------------

set(
        ncnn_DIR
        "${CMAKE_SOURCE_DIR}/ncnn/${ANDROID_ABI}/lib/cmake/ncnn"
)

message(STATUS "======================================")
message(STATUS "ANDROID_ABI = ${ANDROID_ABI}")
message(STATUS "NCNN DIR    = ${ncnn_DIR}")
message(STATUS "======================================")

find_package(ncnn REQUIRED)

# ---------------------------------------------------------
# Native YOLO library
# ---------------------------------------------------------

add_library(
        yolo_native
        SHARED
        native-lib.cpp
)

# ---------------------------------------------------------
# Android native libraries
# ---------------------------------------------------------

# Logcat support
find_library(
        log-lib
        log
)

# Android AssetManager support
find_library(
        android-lib
        android
)

# Android Bitmap access
find_library(
        jnigraphics-lib
        jnigraphics
)

# ---------------------------------------------------------
# Link everything
# ---------------------------------------------------------

target_link_libraries(
        yolo_native

        ncnn

        ${log-lib}
        ${android-lib}
        ${jnigraphics-lib}
)