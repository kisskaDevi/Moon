cmake_minimum_required(VERSION 3.5)

project(cudaRayTracing LANGUAGES CUDA CXX)
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()
set(CMAKE_DEBUG_POSTFIX d)

find_package(CUDA 12.0 REQUIRED)

set(SOURCES hitable/sphere.cu
hitable/triangle.cu
math/vec4.cu
rayTracingGraphics/rayTracingGraphics.cu
rayTracingGraphics/rayTracingLink.cpp
transformational/camera.cu
utils/hitableArray.cu
utils/hitableContainer.cu
utils/hitableList.cu
utils/operations.cu)

set(HEADERS hitable/hitable.h
hitable/sphere.h
hitable/triangle.h
materials/material.h
math/dualQuaternion.h
math/quaternion.h
math/ray.h
math/vec4.h
rayTracingGraphics/rayTracingGraphics.h
rayTracingGraphics/rayTracingLink.h
transformational/camera.h
transformational/object.h
utils/buffer.h
utils/hitableArray.h
utils/hitableContainer.h
utils/hitableList.h
utils/operations.h)

set(CUDA_ARCH compute_86)
set(CUDA_SM sm_86)
set(SYSTEM_TYPE 64)
set(MSVCRT_LINK_FLAG $<$<CONFIG:Debug>:"/MD">
    $<$<CONFIG:Release>:"/MDd">)
set(XFLAGS -Xcompiler "/EHsc,/W3,/nologo,/O2,/FS,${MSVCRT_LINK_FLAG}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../.. ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

#find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
#find_library(CUDA_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
#find_library(CUDADEVRT_LIBRARY cudadevrt ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
#find_library(CUDART_STATIC_LIBRARY lcudart_static ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

#set(CUDA_LIBS ${CUDART_LIBRARY}
#${CUDA_LIBRARY}
#${CUDADEVRT_LIBRARY}
#${CUDART_STATIC_LIBRARY})

add_library(cudaRayTracing STATIC ${SOURCES} ${HEADERS})
target_include_directories(cudaRayTracing PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/hitable
${CMAKE_CURRENT_SOURCE_DIR}/materials
${CMAKE_CURRENT_SOURCE_DIR}/math
${CMAKE_CURRENT_SOURCE_DIR}/rayTracingGraphics
${CMAKE_CURRENT_SOURCE_DIR}/transformational
${CMAKE_CURRENT_SOURCE_DIR}/utils
${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(${PROJECT_NAME}
   PROPERTIES
       CUDA_SEPARABLE_COMPILATION ON
       CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_link_libraries(cudaRayTracing PUBLIC graphicsManager utils ${CUDA_LIBRARIES})
target_compile_options(cudaRayTracing PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -gencode arch=${CUDA_ARCH},code=${CUDA_SM}
                        #-x cu
                        -rdc=true
                        #-std=c++17
                        #--machine ${SYSTEM_TYPE}
                        #--compile
                        #-cudart static
                        #-DNDEBUG -D_CONSOLE -D_UNICODE -DUNICODE
                        #${NVCCFLAGS}
                        #${XFLAGS}
                       >)
