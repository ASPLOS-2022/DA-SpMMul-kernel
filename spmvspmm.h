#include "spmm.h"
#include "spmv.h"
#include "cuda_profiler_api.h"

enum SPMV_SPMM_ALG {
    ALG_CSR_SCALAR,
    ALG_CSR_VECTOR,
    ALG_COO_SCALAR,
    ALG_COO_VECTOR
};


struct SpmvSpmmProblem {
    int nr; // number of sparse matrix rows
    int nc; // number of sparse matrix columns
    int nnz; // number of sparse matrix non-zeros
    int maxNv; // maximum row dimension of dense matrices
    int *rowPtr; // pointer to csr row_offset array
    int *rowIdx; // pointer to coo row_indices array
    int *colIdx; // pointer to csr/coo row_indices array
    float *values; // pointer to csr/coo values array 
    float *dnInput; // pointer to the dense input matrix of nc*maxNv
    float *dnOutput; // pointer to the dense output matrix of size nr*maxNv

    SpmvSpmmProblem(int nr, int nc, int nnz, int maxNv, int *rowPtr, int *rowIdx, int *colIdx, float *values, float *dnInput, float *dnOutput): 
        nr(nr), nc(nc), nnz(nnz), maxNv(maxNv), 
        rowPtr(rowPtr), rowIdx(rowIdx), colIdx(colIdx), values(values), dnInput(dnInput), dnOutput(dnOutput)
    {}

    void run(
        // SparseFormat format, 
        DenseLayout layout, 
        int nv, 
        SPMV_SPMM_ALG alg);
    
    float benchmark(
        DenseLayout layout, 
        int nv, 
        SPMV_SPMM_ALG alg, 
        int warmupIters = 5, 
        int repeatIters = 50);
    
    void dry_run(
        DenseLayout layout, 
        int nv, 
        SPMV_SPMM_ALG alg, 
        int warmupIters = 0
    );
        
};


template<
    int kBlockSize = 256,
    int kWarpSize = 32
>
cudaError_t cudaSpmm(
    SPMV_SPMM_ALG kAlg,
    DenseLayout layout,
    const int nr, 
    const int nc, 
    const int nnz, 
    const int nv, 
    const int *  rowPtr,
    const int *  rowIdx, 
    const int *  colIdx, 
    const float *  values, 
    const float *  dnInput,
    float *dnOutput
)
{
    if (layout == DENSE_ROW_MAJOR)
    {

        if (kAlg == ALG_COO_SCALAR || kAlg == ALG_CSR_SCALAR ) 
        {
            int blockDimX = min(nv, 32);
            int blockDimY = kBlockSize / blockDimX;
            int gridSize = DIV_UP(nr, blockDimY);
            
            if (kAlg == ALG_COO_SCALAR)
            {
                if (nv==1)
                    spmv_coo_scalar_kernel <kBlockSize> <<< gridSize, blockDimX * blockDimY >>> (nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput, dnOutput);
                else if (nv==2)
                    spmm_coo_scalar_row_kernel <kBlockSize, 2> <<< gridSize, blockDimX * blockDimY >>> (nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput, dnOutput);
                else if (nv==4)
                    spmm_coo_scalar_row_kernel <kBlockSize, 4><<< gridSize, blockDimX * blockDimY >>> (nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput, dnOutput);
                else if (nv==8)
                    spmm_coo_scalar_row_kernel <kBlockSize, 8><<< gridSize, blockDimX * blockDimY >>> (nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput, dnOutput);
                else if (nv==16)
                    spmm_coo_scalar_row_kernel <kBlockSize, 16><<< gridSize, blockDimX * blockDimY >>> (nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput, dnOutput);
                else if (nv==32)
                    spmm_coo_scalar_row_kernel <kBlockSize, 32><<< gridSize, blockDimX * blockDimY >>> (nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput, dnOutput);
                else
                    spmm_coo_scalar_row_kernel <kBlockSize, 32><<< dim3(gridSize, nv/32, 1), blockDimX * blockDimY >>> (nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput, dnOutput);
            }
            else // ALG_CSR_SCALAR
            {
                if (nv==1)
                    spmv_csr_scalar_kernel <kBlockSize> <<< gridSize, blockDimX * blockDimY >>> (nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput, dnOutput);
                else if (nv==2)
                    spmm_csr_scalar_row_kernel <kBlockSize, 2> <<< gridSize, blockDimX * blockDimY >>> (nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput, dnOutput);
                else if (nv==4)
                    spmm_csr_scalar_row_kernel <kBlockSize, 4><<< gridSize, blockDimX * blockDimY >>> (nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput, dnOutput);
                else if (nv==8)
                    spmm_csr_scalar_row_kernel <kBlockSize, 8><<< gridSize, blockDimX * blockDimY >>> (nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput, dnOutput);
                else if (nv==16)
                    spmm_csr_scalar_row_kernel <kBlockSize, 16><<< gridSize, blockDimX * blockDimY >>> (nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput, dnOutput);
                else if (nv==32)
                    spmm_csr_scalar_row_kernel <kBlockSize, 32><<< gridSize, blockDimX * blockDimY >>> (nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput, dnOutput);
                else 
                    spmm_csr_scalar_row_kernel <kBlockSize, 32><<< dim3(gridSize, (nv/32), 1), blockDimX * blockDimY >>> (nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput, dnOutput);
            }
        }
        else if (kAlg == ALG_COO_VECTOR || kAlg == ALG_CSR_VECTOR ) {

            int blockDimX = kWarpSize;
            int blockDimY = kBlockSize / kWarpSize;
            int numberOfElemLoad = nv > 4 ? 4 : nv;
            int numberOfWarps = min( DIV_UP(nnz, kWarpSize), nr) * (min(nv, 32) / numberOfElemLoad);
            int gridSize = numberOfWarps / blockDimY;
            
            if (kAlg == ALG_COO_VECTOR)
            {
                if (nv==1)
                    spmv_coo_vector_kernel < kBlockSize, float><<<gridSize, blockDimX * blockDimY >>> (nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput, dnOutput);
                else if (nv==2)
                    spmv_coo_vector_kernel < kBlockSize, float2><<<gridSize, blockDimX * blockDimY >>> (nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput, dnOutput);
                else if (nv==4)
                    spmv_coo_vector_kernel < kBlockSize, float4><<<gridSize, blockDimX * blockDimY >>> (nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput, dnOutput);
                else if (nv<=32)
                    spmm_coo_vector_row_kernel < kBlockSize, float4><<<gridSize, blockDimX * blockDimY >>> (nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput, dnOutput);
                else 
                    spmm_coo_vector_row_kernel < kBlockSize, float4><<<dim3(gridSize, (nv/32), 1), blockDimX * blockDimY >>> (nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput, dnOutput);
            }
            else // ALG_CSR_VECTOR 
            {
                if (nv==1)
                    spmv_csr_vector_kernel < kBlockSize, float><<<gridSize, blockDimX * blockDimY >>> (nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput, dnOutput);
                else if (nv==2)
                    spmv_csr_vector_kernel < kBlockSize, float2><<<gridSize, blockDimX * blockDimY >>> (nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput, dnOutput);
                else if (nv==4)
                    spmv_csr_vector_kernel < kBlockSize, float4><<<gridSize, blockDimX * blockDimY >>> (nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput, dnOutput);
                else if (nv<=32)
                    spmm_csr_vector_row_kernel < kBlockSize, float4><<<gridSize, blockDimX * blockDimY >>> (nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput, dnOutput);
                else
                    spmm_csr_vector_row_kernel < kBlockSize, float4><<<dim3(gridSize, (nv/32), 1), blockDimX * blockDimY >>> (nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput, dnOutput);
            }
        }

    }
    else 
    {
        if (kAlg == ALG_COO_SCALAR || kAlg == ALG_CSR_SCALAR ) 
        {            
            if (kAlg == ALG_COO_SCALAR)
            {
                if (nv==1)
                    spmv_coo_scalar_kernel <kBlockSize> <<< (DIV_UP(nr, kBlockSize)), kBlockSize >>> (nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput, dnOutput);
                else 
                    spmm_coo_scalar_col_kernel <kBlockSize, 1> <<< dim3((DIV_UP(nr, kBlockSize)), nv, 1), kBlockSize >>>(nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput, dnOutput);
            }
            else // ALG_CSR_SCALAR
            {
                if (nv==1)
                    spmv_csr_scalar_kernel <kBlockSize> <<< (DIV_UP(nr, kBlockSize)), kBlockSize >>> (nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput, dnOutput);
                else 
                    spmm_csr_scalar_col_kernel <kBlockSize, 1> <<< dim3((DIV_UP(nr, kBlockSize)), nv, 1), kBlockSize >>>(nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput, dnOutput);
            }
        }
        else if (kAlg == ALG_COO_VECTOR || kAlg == ALG_CSR_VECTOR ) {

            int blockDimX = kWarpSize;
            int blockDimY = kBlockSize / kWarpSize;
            int numberOfElemLoad = 1;
            int numberOfWarps = min( DIV_UP(nnz, kWarpSize), nr) * (nv / numberOfElemLoad);
            int gridSize = numberOfWarps / blockDimY;
            int gridDimX = min( DIV_UP(nnz, kWarpSize), nr) / blockDimY;
            int gridDimY = nv;
            
            if (kAlg == ALG_COO_VECTOR)
            {
                if (nv==1)
                    spmv_coo_vector_kernel < kBlockSize, float><<<gridSize, blockDimX * blockDimY >>> (nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput, dnOutput);
                else 
                    spmm_coo_vector_col_kernel<kBlockSize, 1> <<< dim3((DIV_UP(nnz, (kBlockSize/32))), nv, 1), kBlockSize >>>(nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput, dnOutput);
            }
            else // ALG_CSR_VECTOR 
            {
                if (nv==1)
                    spmv_csr_vector_kernel < kBlockSize, float><<<gridSize, blockDimX * blockDimY >>> (nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput, dnOutput);
                else 
                    spmm_csr_vector_col_kernel <kBlockSize, 1><<< dim3((DIV_UP(nr, (kBlockSize/32))), nv, 1), kBlockSize >>>(nr, nc, nnz, nv, rowPtr, rowIdx, colIdx, values, dnInput, dnOutput);
            }
        }
    }
    return cudaPeekAtLastError();
}


void SpmvSpmmProblem::run(
    // SparseFormat format, 
    DenseLayout layout, 
    int nv, 
    SPMV_SPMM_ALG alg)
{
    // if ( (nv & (nv-1)) != 0 || nv>32 ) // not power of 2
    if ( (nv & (nv-1)) != 0 ) // not power of 2
    {
        std::cerr << "nv must be power of 2, less than 32. Got nv = " << nv << std::endl;
        return ;
    }

    // run Spmm algorithm   
    cudaError_t status = cudaSpmm<>(alg, layout, this->nr, this->nc, this->nnz, nv, this->rowPtr, this->rowIdx, this->colIdx, this->values, this->dnInput, this->dnOutput);
    CUDA_CHECK( status );    
}

float SpmvSpmmProblem::benchmark(DenseLayout layout, int nv, SPMV_SPMM_ALG alg, int warmupIters, int repeatIters)
{
    // if ( (nv & (nv-1)) != 0 || nv>32 ) // not power of 2
    if ( (nv & (nv-1)) != 0 ) // not power of 2
    {
        std::cerr << "nv must be power of 2, less than 32. Got nv = " << nv << std::endl;
        return -1;
    }
    cudaError_t status;
    hgpu::GpuTimer gpuTimer;
    // run spmm benchmarks
    for (int iteration=0; iteration < warmupIters; iteration++) {
        status = cudaSpmm<>(alg, layout, this->nr, this->nc, this->nnz, nv, this->rowPtr, this->rowIdx, this->colIdx, this->values, this->dnInput, this->dnOutput);
    }
    CUDA_CHECK( status );
    CUDA_CHECK( cudaDeviceSynchronize());
    gpuTimer.start();
    for (int iteration=0; iteration < repeatIters; iteration++) {
        status = cudaSpmm<>(alg, layout, this->nr, this->nc, this->nnz, nv, this->rowPtr, this->rowIdx, this->colIdx, this->values, this->dnInput, this->dnOutput);
    }
    gpuTimer.stop();
    CUDA_CHECK( status );    
    return (gpuTimer.elapsed_msecs()/((float)repeatIters));
}

void SpmvSpmmProblem::dry_run(
    DenseLayout layout, 
    int nv, 
    SPMV_SPMM_ALG alg, 
    int warmupIters
)
{
    if ( (nv & (nv-1)) != 0 || nv>32 ) // not power of 2
    {
        std::cerr << "nv must be power of 2, less than 32. Got nv = " << nv << std::endl;
        return;
    }
    cudaError_t status;
    // run spmm benchmarks
    for (int iteration=0; iteration < warmupIters; iteration++) {
        status = cudaSpmm<>(alg, layout, this->nr, this->nc, this->nnz, nv, this->rowPtr, this->rowIdx, this->colIdx, this->values, this->dnInput, this->dnOutput);
    }
    CUDA_CHECK( status );
    CUDA_CHECK( cudaDeviceSynchronize());
#ifndef FLAG_GPGPUSIM
    cudaProfilerStart();
#endif // FLAG_GPGPUSIM
        status = cudaSpmm<>(alg, layout, this->nr, this->nc, this->nnz, nv, this->rowPtr, this->rowIdx, this->colIdx, this->values, this->dnInput, this->dnOutput);
#ifndef FLAG_GPGPUSIM
    cudaProfilerStop();
#endif // FLAG_GPGPUSIM

    CUDA_CHECK( status );    
}
