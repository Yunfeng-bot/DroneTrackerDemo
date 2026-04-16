cmake_minimum_required(VERSION 3.22.1)
project(dronetracker_native)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(
    dronetracker_native
    SHARED
    jni/NativeTrackerBridge.cpp
    tracker/NanoTrackerEngine.cpp
    tracker/NcnnTrackerImpl.cpp
    tracker/RknnTrackerImpl.cpp
)

target_include_directories(
    dronetracker_native
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set(NCNN_ANDROID_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/ncnn/android/ncnn-20260113-android")
set(NCNN_ABI_ROOT "${NCNN_ANDROID_ROOT}/${ANDROID_ABI}")
set(NCNN_ABI_INCLUDE "${NCNN_ABI_ROOT}/include")
set(NCNN_ABI_CMAKE_DIR "${NCNN_ABI_ROOT}/lib/cmake/ncnn")

if(EXISTS "${NCNN_ABI_INCLUDE}/ncnn/net.h" AND EXISTS "${NCNN_ABI_CMAKE_DIR}/ncnnConfig.cmake")
    set(ncnn_DIR "${NCNN_ABI_CMAKE_DIR}")
    find_package(ncnn CONFIG QUIET)
endif()

if(ncnn_FOUND AND TARGET ncnn)
    message(STATUS "dronetracker_native: enable ncnn for ABI ${ANDROID_ABI} from ${NCNN_ANDROID_ROOT}")
    target_compile_definitions(dronetracker_native PRIVATE DRONETRACKER_HAVE_NCNN=1)
    target_include_directories(dronetracker_native PRIVATE ${NCNN_ABI_INCLUDE})
    target_link_libraries(dronetracker_native ncnn)
else()
    message(WARNING "dronetracker_native: ncnn package missing for ABI ${ANDROID_ABI}, build without real ncnn runtime")
    target_compile_definitions(dronetracker_native PRIVATE DRONETRACKER_HAVE_NCNN=0)
endif()

find_library(log-lib log)
find_library(android-lib android)

target_link_libraries(
    dronetracker_native
    ${log-lib}
    ${android-lib}
)
