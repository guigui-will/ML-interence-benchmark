cmake_minimum_required(VERSION 3.18)
project(ml_inference_benchmark LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)  # CUDA::cudart, CUDA::nvml

set(ORT_ROOT "" CACHE PATH "Raiz do ONNX Runtime GPU (include/ e lib/)")
set(TENSORRT_ROOT "/usr" CACHE PATH "Raiz do TensorRT")

add_library(bench_common INTERFACE)
target_include_directories(bench_common INTERFACE cpp/common)
target_link_libraries(bench_common INTERFACE CUDA::nvml)

# ---- ONNX Runtime ----
if(ORT_ROOT)
  add_executable(ort_bench cpp/ort_bench/main.cpp)
  target_include_directories(ort_bench PRIVATE ${ORT_ROOT}/include)
  target_link_directories(ort_bench PRIVATE ${ORT_ROOT}/lib)
  target_link_libraries(ort_bench PRIVATE bench_common onnxruntime)
  set_target_properties(ort_bench PROPERTIES BUILD_RPATH ${ORT_ROOT}/lib)
else()
  message(STATUS "ORT_ROOT não definido: ort_bench não será compilado")
endif()

# ---- TensorRT ----
find_path(TRT_INCLUDE NvInfer.h HINTS ${TENSORRT_ROOT}/include ${TENSORRT_ROOT}/include/x86_64-linux-gnu)
find_library(TRT_LIB nvinfer HINTS ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib/x86_64-linux-gnu)
if(TRT_INCLUDE AND TRT_LIB)
  add_executable(trt_bench cpp/trt_bench/main.cpp)
  target_include_directories(trt_bench PRIVATE ${TRT_INCLUDE})
  target_link_libraries(trt_bench PRIVATE bench_common CUDA::cudart ${TRT_LIB})
else()
  message(STATUS "TensorRT não encontrado: trt_bench não será compilado")
endif()
