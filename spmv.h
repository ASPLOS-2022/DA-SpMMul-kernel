#include "helper.h"

template<
    int kCtaSize
>
__global__ void spmv_csr_scalar_kernel
(
    const int nr, 
    const int nc, 
    const int nnz, 
    const int nv, 
    const int * __restrict__ rowPtr,
    const int * __restrict__ rowIdx, 
    const int * __restrict__ colIdx, 
    const float * __restrict__ values, 
    const float * __restrict__ dnInput,
    float *dnOutput
)
{
    // if (blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0)
    // {
    //     printf("spmv_csr_scalar_kernel");
    // }
    int tid = kCtaSize * blockIdx.x + threadIdx.x;
    int stride = kCtaSize * gridDim.x;
    float val, res = 0; int col;
    for (int row = tid; row < nr; row += stride)
    {
        int start = __ldg(rowPtr + row);
        int end = __ldg(rowPtr + row + 1);
        for ( int p=start; p<end; p++ )
        {
            col = __ldg(colIdx + p);
            val = __ldg(values + p);
            res += val * __ldg(dnInput + col);
        }
        dnOutput[row] = res;
    }
}

template<
    int kCtaSize
>
__global__ void spmv_coo_scalar_kernel
(
    const int nr, 
    const int nc, 
    const int nnz, 
    const int nv, 
    const int * __restrict__ rowPtr,
    const int * __restrict__ rowIdx, 
    const int * __restrict__ colIdx, 
    const float * __restrict__ values, 
    const float * __restrict__ dnInput,
    float *dnOutput
)
{
    // if (blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0)
    // {
    //     printf("spmv_coo_scalar_kernel");
    // }
    int tid = kCtaSize * blockIdx.x + threadIdx.x;
    int NE_PER_THREAD = DIV_UP(nnz, (gridDim.x*kCtaSize));
    tid *= NE_PER_THREAD;
    
    if (tid < nnz) {

        int row = __ldg(rowIdx + tid);
        int col = __ldg(colIdx + tid);
        float val = __ldg(values + tid) * __ldg(dnInput + col);
        int curr_row = row;

        for (int ii = 1; ii < NE_PER_THREAD && ++tid < nnz; ii++) {

            row = __ldg(rowIdx + tid);
            col = __ldg(colIdx + tid);
            
            if (row!=curr_row) {
                atomicAdd(&dnOutput[curr_row], val);
                val = __ldg(values + tid) * __ldg(dnInput + col);
                curr_row = row;
            }
            else {
                val += __ldg(values + tid) * __ldg(dnInput + col);
            }
        }
        atomicAdd(&dnOutput[curr_row], val);       
    }
}

template<
    int kCtaSize,
    typename LoadStoreType
>
__global__ void spmv_csr_vector_kernel 
(
    const int nr, 
    const int nc, 
    const int nnz, 
    const int nv, 
    const int * __restrict__ rowPtr,
    const int * __restrict__ rowIdx, 
    const int * __restrict__ colIdx, 
    const float * __restrict__ values, 
    const float * __restrict__ dnInput,
    float *dnOutput
)
{
    // if (blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0)
    // {
    //     printf("spmv_csr_vector_kernel");
    // }
    constexpr int numberOfElements = sizeof(LoadStoreType) / sizeof(float);
    
    int tid = kCtaSize * blockIdx.x + threadIdx.x;
    int warp_id = tid >> 5;
    int stride = (kCtaSize * gridDim.x) >> 5;
    int lane_id = tid & (32-1);
    for (int row = warp_id; row < nr; row += stride) {
        // get row offsets
        int start = __ldg(rowPtr + row);
        int end = __ldg(rowPtr + row + 1);
        // float res = 0.0, val;
        float res[numberOfElements] = {0};
        float dnVal[numberOfElements];
        float val; int col;
        
        for (int jj = start + lane_id; jj < end; jj += 32) {
            col = __ldg(colIdx + jj);
            val = __ldg(values + jj);
            ldg_float<LoadStoreType>(dnVal, dnInput + col*nv );
            #pragma unroll
            for (int kk=0; kk<numberOfElements; kk++) 
                res[kk] += val * dnVal[kk];
        }

        #pragma unroll
        for (int kk=0; kk<numberOfElements; kk++)
        {
            SHFL_DOWN_REDUCE(res[kk])
        }
        
        if (lane_id == 0) {
            st_float<LoadStoreType>( (dnOutput + row*nv), res);
        }
    }
}

template<
    int kBlockSize,
    typename LoadStoreType
>
__global__ void spmv_coo_vector_kernel 
(
    const int nr, 
    const int nc, 
    const int nnz, 
    const int nv, 
    const int * __restrict__ rowPtr,
    const int * __restrict__ rowIdx, 
    const int * __restrict__ colIdx, 
    const float * __restrict__ values, 
    const float * __restrict__ dnInput,
    float *dnOutput
)
{
    // if (blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0)
    // {
    //     printf("spmv_coo_vector_kernel");
    // }
    constexpr int numberOfElements = sizeof(LoadStoreType) / sizeof(float);
    
    int tid = kBlockSize * blockIdx.x + threadIdx.x;
    int lane_id = tid & (32-1);
    int stride = kBlockSize * gridDim.x;
    int row, col = 0; float val = 0.0;
    for (; tid < nnz + lane_id; tid += stride) {
        float res[numberOfElements] = {0};

        if (tid < nnz) {

            row = __ldg(rowIdx + tid);
            col = __ldg(colIdx + tid);
            val = __ldg(values + tid);
            ldg_float<LoadStoreType>(res, dnInput + col*nv);
            #pragma unroll
            for (int kk=0; kk<numberOfElements; kk++) 
                res[kk] *= val;
            

        }
        else {
            row = nr-1;
            col = 0;
            #pragma unroll
            for (int kk=0; kk<numberOfElements; kk++) 
                res[kk] = 0;
        }
        
        int row_intv = __shfl_sync(FULLMASK, row, (32-1)) - __shfl_sync(FULLMASK, row, 0);
        if (row_intv == 0) {
            #pragma unroll 
            for (int kk=0; kk<numberOfElements; kk++)
            {
                SHFL_DOWN_REDUCE(res[kk]);
            }
            if (lane_id==0) 
            {
                #pragma unroll 
                for (int kk=0; kk<numberOfElements; kk++)
                    atomicAdd(&dnOutput[row*nv + kk], res[kk]);
                
            }
        }
        else {
            bool is_seg_start = ((__shfl_up_sync(FULLMASK, row, 1) != row) || (lane_id == 0));
            float tmpv; int tmpr;
            #pragma unroll 
            for (int kk=0; kk<numberOfElements; kk++)
            {
                SEG_SHFL_SCAN(res[kk], tmpv, row, tmpr);
            }
            if (is_seg_start) {
                
                #pragma unroll 
                for (int kk=0; kk<numberOfElements; kk++)
                    atomicAdd(&dnOutput[row*nv + kk], res[kk]);
            }
        }
    }
}
