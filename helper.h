#ifndef SPMV_SPMM_HELPER
#define SPMV_SPMM_HELPER

#include <stddef.h>
#include <stdint.h>
#include <stdexcept>
#include <vector>
#include <cassert>
#include <iostream>
#include <math.h>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"


#ifndef __NVCC__
#define __syncthreads()

#endif

//#define FLAG_GPGPUSIM

#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__                             \
                << " of file " << __FILE__ << std::endl;                \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

namespace hgpu 
{

// I feel the logic about randomization is wrong....need to think about random seed carefully.

template<typename DType>
struct InitGen {
    uint seed;
    // DType mean;
    InitGen():seed(time(0)) {}
    InitGen(uint seed):seed(seed) {}
    DType gen_next() 
    {
        std::cerr << "Random generator in abstract type is not implemented. Use <int> or <float> specialization. \n";
        return (DType)0;
    }
};

template<>
struct InitGen<int>
{
    uint seed;
    InitGen() { seed = time(0); srand(seed); }
    InitGen(uint seed): seed(seed) { srand(seed); }
    int gen_next()
    {
        return ((rand() % 100) - 50);
    }
};

template<>
struct InitGen<float>
{
    uint seed; 
    InitGen() { seed = time(0); srand(seed); }
    InitGen(uint seed): seed(seed) { srand(seed); }
    float gen_next()
    {
        return (float)(rand() %1000 - 500)/1000;
    }
};

template<typename DType>
class CUArray {

private:

std::vector<DType> host_array;
DType *device_data = NULL;
int size_=0;

public:
void free_() 
{
    if (device_data != NULL)
    {
        CUDA_CHECK(cudaFree(device_data));
        device_data=NULL;
    }
}

CUArray(): size_(0), device_data(NULL) {}
CUArray(std::vector<DType>& data)
{
    size_ = data.size();
    host_array = data;
    device_data = NULL ;
}

~CUArray() { free_(); }

void init_random(int size)
{
    host_array = std::vector<DType>(size);

    InitGen<DType> generator;
    for (int i=0; i<size; i++)
        host_array[i] = generator.gen_next();

    size_ = host_array.size();
}

void init_zeros(int size)
{
    host_array = std::vector<DType>(size, (DType)0);
    size_ = host_array.size();
}
void assign(std::vector<DType>& data)
{
    size_ = data.size();
    host_array = data;
    
    if (device_data != NULL) {
        // already allocated
        free_();
        device_data = NULL;
    }
}

void sync_device() 
{
    free_();
    CUDA_CHECK(cudaMalloc((void**)&device_data, size_ * sizeof(DType)));
    CUDA_CHECK(cudaMemcpy(device_data, host_array.data(), size_ * sizeof(DType), cudaMemcpyHostToDevice));
}

std::vector<DType> sync_host()
{
    assert((size_ == host_array.size()));
    CUDA_CHECK(cudaMemcpy(host_array.data(), device_data, size_ * sizeof(DType), cudaMemcpyDeviceToHost));
    return host_array;
}

DType* get_device_data() { return device_data; }
int get_size() { return size_; }
DType* get_host_data() { return host_array.data(); }

std::vector<DType> &get_host_array() { return host_array; }
};


// credit to cub library.
struct GpuTimer
{
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;

    GpuTimer() {
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
    }

    ~GpuTimer() 
    {
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
    }

    void start()
    {
        cudaEventRecord(startEvent, 0);
    }

    void stop()
    {
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
    }

    float elapsed_msecs()
    {
        float elapsed;
        cudaEventElapsedTime(&elapsed, startEvent, stopEvent);
        return elapsed;
    }
};

}

/////////////////////////////////////////////
// Utilities for data loading
/////////////////////////////////////////////
// __device__ __forceinline__ float2 ldg_float2(const float* a)
// {
//     const float2 *aa = reinterpret_cast<const float2 *>(a);
//     float2 r = __ldg(aa);
//     // asm volatile ("ld.global.nc.v2.f32 {%0,%1}, [%2];" :
//                     // "=f"(r.x), "=f"(r.y) : "l"(a));
//     return r;
// }
// __device__ __forceinline__ float4 ldg_float4(const float* a)
// {
//     const float4 *aa = reinterpret_cast<const float4 *>(a);
//     float4 r = __ldg(aa);
//     // asm volatile ("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];" :
//     //                 "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)  : "l"(a));
//     return r;
// }
template<typename T>
__device__ __forceinline__ void ldg_float(float *r, const float*a)
{
    (reinterpret_cast<T *>(r))[0] = *(reinterpret_cast<const T*>(a));
}
template<typename T>
__device__ __forceinline__ void st_float(float *a, float *v)
{
    *(T*)a = *(reinterpret_cast<T *>(v));
}
// __device__ __forceinline__ void init_float2(float2 r)
// {
//     asm volatile ("mov.b32 %0,0;\n mov.b32 %1,0;" : "=f"(r.x), "=f"(r.y));
// }
// __device__ __forceinline__ void init_float4(float4 r)
// {
//     asm volatile ("mov.b32 %0,0;\n mov.b32 %1,0;\n mov.b32 %0,0;\n mov.b32 %1,0;" 
//     : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w));
// }
// __device__ __forceinline__ void st_float2(float *a, float2 v)
// {
//     *(float2*)a = v;
// }
// __device__ __forceinline__ void st_float4(float *a, float4 v)
// {
//     *(float4*)a = v;
// }
__device__ __forceinline__ void mac_float2(float4 c, const float a, const float2 b)
{
    c.x += a * b.x ; c.y += a * b.y ;
}
__device__ __forceinline__ void mac_float4(float4 c, const float a, const float4 b)
{
    c.x += a * b.x ; c.y += a * b.y ; c.z += a * b.z ; c.w += a * b.w ; 
}


//////////////////////////////////////////////
// specialization for spmvspmm
//////////////////////////////////////////////

// utilities for sparse matrix operations
enum SparseFormat {
    SPARSE_FORMAT_CSR,
    SPARSE_FORMAT_COO
};

enum DenseLayout {
    DENSE_ROW_MAJOR,
    DENSE_COL_MAJOR
};

#define FULLMASK 0xffffffff
#define DIV_UP(x,y) (((x)+(y)-1)/(y))

#define SHFL_DOWN_REDUCE(v) \
v += __shfl_down_sync(FULLMASK, v, 16);\
v += __shfl_down_sync(FULLMASK, v, 8);\
v += __shfl_down_sync(FULLMASK, v, 4);\
v += __shfl_down_sync(FULLMASK, v, 2);\
v += __shfl_down_sync(FULLMASK, v, 1);

#define SEG_SHFL_SCAN(v, tmpv, segid, tmps) \
tmpv = __shfl_down_sync(FULLMASK, v, 1); tmps = __shfl_down_sync(FULLMASK, segid, 1); if (tmps == segid && lane_id < 31) v += tmpv;\
tmpv = __shfl_down_sync(FULLMASK, v, 2); tmps = __shfl_down_sync(FULLMASK, segid, 2); if (tmps == segid && lane_id < 30) v += tmpv;\
tmpv = __shfl_down_sync(FULLMASK, v, 4); tmps = __shfl_down_sync(FULLMASK, segid, 4); if (tmps == segid && lane_id < 28) v += tmpv;\
tmpv = __shfl_down_sync(FULLMASK, v, 8); tmps = __shfl_down_sync(FULLMASK, segid, 8); if (tmps == segid && lane_id < 24) v += tmpv;\
tmpv = __shfl_down_sync(FULLMASK, v, 16); tmps = __shfl_down_sync(FULLMASK, segid, 16); if (tmps == segid && lane_id < 16) v += tmpv;

// template <typename index_t>
// __device__ __forceinline__ index_t binary_search_segment_number(
//     index_t *seg_offsets, index_t n_seg, index_t n_elem, index_t elem_id
// ) 
// {
//     // this function finds the first element in seg_offsets greater than elem_id (n^th)
//     // and returns n-1
//     index_t lo = 1, hi = n_seg, mid;
//     while (lo < hi) {
//         mid = (lo + hi) >> 1;
//         if (seg_offsets[mid] <= elem_id) {
//             lo = mid + 1;
//         }
//         else {
//             hi = mid;
//         }
//     }
//     return (hi - 1);
// }

// template <typename index_t>
// __device__ __forceinline__ index_t binary_search_segment_number(
//     const index_t *seg_offsets, index_t n_seg, index_t n_elem, index_t elem_id
// ) 
// {
//     // this function finds the first element in seg_offsets greater than elem_id (n^th)
//     // and returns n-1
//     index_t lo = 1, hi = n_seg, mid;
//     while (lo < hi) {
//         mid = (lo + hi) >> 1;
//         if (seg_offsets[mid] <= elem_id) {
//             lo = mid + 1;
//         }
//         else {
//             hi = mid;
//         }
//     }
//     return (hi - 1);
// }

#endif // SPMV_SPMM_HELPER