#include "helper.h"
#include "cusparse_v2.h"

#define CUSPARSE_CHECK(func)    \
{                                                                           \
    cusparseStatus_t status = ( func );                                     \
    if (status != CUSPARSE_STATUS_SUCCESS)                                  \
    {                                                                       \
        std::cerr << "Got cusparse error"                                   \
                << " at line: " << __LINE__                                 \
                << " of file " << __FILE__ << std::endl;                    \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
}


#if CUDA_VERSION < 11000
struct CusparseCsrmmProblem
{
    cusparseHandle_t handle;
    cusparseMatDescr_t descr;

    float alpha;
    float beta;
    int nr; // number of sparse matrix rows
    int nc; // number of sparse matrix columns
    int nnz; // number of sparse matrix non-zeros
    int maxNv; // maximum row dimension of dense matrices
    int *rowPtr; // pointer to csr row_offset array
    int *colIdx; // pointer to csr row_indices array
    float *values; // pointer to csr values array 
    float *dnInput; // pointer to the dense input matrix of nc*maxNv
    float *dnOutput; // pointer to the dense output matrix of size nr*maxNv

    CusparseCsrmmProblem(int nr, int nc, int nnz, int maxNv, int *rowPtr, int *colIdx, float *values, float *dnInput, float *dnOutput): 
    nr(nr), nc(nc), nnz(nnz), maxNv(maxNv), 
    rowPtr(rowPtr), colIdx(colIdx), values(values), dnInput(dnInput), dnOutput(dnOutput),
    alpha(1.0), beta(0.0) 
    {
        CUSPARSE_CHECK(cusparseCreate(&handle));
        CUSPARSE_CHECK(cusparseCreateMatDescr(&descr));
        CUSPARSE_CHECK(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
        CUSPARSE_CHECK(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
    }

    ~CusparseCsrmmProblem() {
        CUSPARSE_CHECK(cusparseDestroy(handle));
        CUSPARSE_CHECK(cusparseDestroyMatDescr(descr));
    }

    cusparseStatus_t _bareRunCsrmv()
    {
        return cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, nr, nc, nnz, &alpha, descr, values, rowPtr, colIdx, dnInput, &beta, dnOutput);
    }
    cusparseStatus_t _bareRunCsrmm2(cusparseOperation_t denseOp, int ldDense, int nv) {
        return cusparseScsrmm2(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, denseOp, nr, nv, nc, nnz, &alpha, descr, values, rowPtr, colIdx, dnInput, ldDense, &beta, dnOutput, nr);
    }

    void run(DenseLayout layout, int nv ) {
        if (nv>maxNv) {
            std::cout << "Got nv = " << nv << " but maxNv = " << maxNv << "." << std::endl;
            return;
        }
        if (nv==1) { // launch csrmv
            CUSPARSE_CHECK( _bareRunCsrmv() );
        }
        else {
            // dense matrix layout
            cusparseOperation_t denseOp = (layout == DENSE_COL_MAJOR ) ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE; 
            
            // dense matrix leading dimension
            int ldDense = (layout == DENSE_COL_MAJOR) ? nc : nv;

            CUSPARSE_CHECK( _bareRunCsrmm2( denseOp, ldDense, nv ));
        }
    }

    float benchmark(DenseLayout layout, int nv, int repeatIters = 200, int warmupIters = 20)
    {
        if (nv>maxNv) {
            std::cout << "Got nv = " << nv << " but maxNv = " << maxNv << "." << std::endl;
            return -1;
        }
        hgpu::GpuTimer gpuTimer;
        if (nv==1) {
            for (int iteration=0; iteration < warmupIters; iteration++) {
                _bareRunCsrmv();
            }
            CUDA_CHECK( cudaPeekAtLastError());
            CUDA_CHECK( cudaDeviceSynchronize());
            gpuTimer.start();
            for (int iteration=0; iteration < repeatIters; iteration++) {
                _bareRunCsrmv();
            }
            gpuTimer.stop();
            return (gpuTimer.elapsed_msecs()/((float)repeatIters));
        }
        else {
            // dense matrix layout
            cusparseOperation_t denseOp = (layout == DENSE_COL_MAJOR ) ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE; 
            
            // dense matrix leading dimension
            int ldDense = (layout == DENSE_COL_MAJOR) ? nc : nv;
            for (int iteration=0; iteration < warmupIters; iteration++) {
                _bareRunCsrmm2( denseOp, ldDense, nv );
            }
            CUDA_CHECK( cudaPeekAtLastError());
            CUDA_CHECK( cudaDeviceSynchronize());
            gpuTimer.start();
            for (int iteration=0; iteration < repeatIters; iteration++) {
                _bareRunCsrmm2( denseOp, ldDense, nv );
            }
            gpuTimer.stop();
            return (gpuTimer.elapsed_msecs()/((float)repeatIters));
        }
    }
};

#else 
struct CusparseSpmmProblem
{
    cusparseHandle_t handle;
    cusparseSpMatDescr_t csrDescr;
    cusparseSpMatDescr_t cooDescr;
    cusparseDnVecDescr_t dnVecInputDescr;
    cusparseDnVecDescr_t dnVecOutputDescr;
    
    float alpha;
    float beta;
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

    float *workspace; // the workspace buffer is cleared every time run() or benchmark() finishes.

    CusparseSpmmProblem(int nr, int nc, int nnz, int maxNv, int *rowPtr, int *rowIdx, int *colIdx, float *values, float *dnInput, float *dnOutput): 
    nr(nr), nc(nc), nnz(nnz), maxNv(maxNv), 
    rowPtr(rowPtr), rowIdx(rowIdx), colIdx(colIdx), values(values), dnInput(dnInput), dnOutput(dnOutput),
    alpha(1.0), beta(0.0)
    {
        CUSPARSE_CHECK(cusparseCreate(&handle));

        // creating sparse csr matrix
        CUSPARSE_CHECK(cusparseCreateCsr(&csrDescr, 
            nr, nc, nnz, rowPtr, colIdx, values, 
            CUSPARSE_INDEX_32I, // index 32-integer for rowptr
            CUSPARSE_INDEX_32I, // index 32-integer for colidx
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F // datatype: 32-bit float real number
        ));

        // creating sparse coo matrix
        CUSPARSE_CHECK(cusparseCreateCoo(&cooDescr, 
            nr, nc, nnz, rowIdx, colIdx, values, 
            CUSPARSE_INDEX_32I, // index 32-integer for rowptr
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F // datatype: 32-bit float real number
        ));

        // creating dense vectors
        CUSPARSE_CHECK(cusparseCreateDnVec(&dnVecInputDescr,
            nc, // input vector length is number of columns
            dnInput,  // we always reuse the same dense matrix space to save space
            CUDA_R_32F // vector's datatype
        ));
        CUSPARSE_CHECK(cusparseCreateDnVec(&dnVecOutputDescr,
            nr, // output vector length is number of rows
            dnOutput,  // we always reuse the same dense matrix space to save space
            CUDA_R_32F // vector's datatype
        ));

    }
    // todo: destroy 

    size_t getWorkspaceSize(SparseFormat format, cusparseSpMVAlg_t alg)
    {
        // TODO: check matching of format & algorithm like in SPMM workspace function

        size_t space;
        CUSPARSE_CHECK( cusparseSpMV_bufferSize(
                            handle, 
                            CUSPARSE_OPERATION_NON_TRANSPOSE, 
                            &alpha, 
                            (format == SPARSE_FORMAT_CSR) ? csrDescr : cooDescr,
                            dnVecInputDescr, 
                            &beta, 
                            dnVecOutputDescr, 
                            CUDA_R_32F, 
                            alg, 
                            &space) );
        return space;
    }

    cusparseStatus_t _bareRunSpMV(SparseFormat format, cusparseSpMVAlg_t alg)
    {
        return cusparseSpMV(
                        handle, 
                        CUSPARSE_OPERATION_NON_TRANSPOSE, 
                        &alpha, 
                        (format == SPARSE_FORMAT_CSR) ? csrDescr : cooDescr,
                        dnVecInputDescr, 
                        &beta,
                        dnVecOutputDescr,
                        CUDA_R_32F,
                        alg,
                        workspace);
    }

    void run(SparseFormat format, DenseLayout layout, int nv, cusparseSpMVAlg_t alg) {
        if (nv!=1) {
            std::cout << "Got nv = " << nv << " but got spmv algorithm.\n";
            assert(0);
        }
        
        // allocate workspace on device
        size_t workspaceSize = getWorkspaceSize(format, alg);
        CUDA_CHECK( cudaMalloc(&workspace, workspaceSize ));

        // run spmv
        CUSPARSE_CHECK( _bareRunSpMV( format, alg ) );

        // wipe memory
        CUDA_CHECK( cudaFree(workspace) );
    }

    float benchmark(SparseFormat format, DenseLayout layout, int nv, cusparseSpMVAlg_t alg, int repeatIters = 200, int warmupIters = 20)
    {
        if (nv!=1) {
            std::cout << "Got nv = " << nv << " but got spmv algorithm.\n";
            assert(0);
        }
        // allocate workspace on device
        size_t workspaceSize = getWorkspaceSize(format, alg);
        CUDA_CHECK( cudaMalloc(&workspace, workspaceSize ));

        hgpu::GpuTimer gpuTimer;
        
        for (int iteration=0; iteration < warmupIters; iteration++) {
             _bareRunSpMV( format, alg );

        }
        CUDA_CHECK( cudaPeekAtLastError());
        CUDA_CHECK( cudaDeviceSynchronize());

        gpuTimer.start();
        for (int iteration=0; iteration < repeatIters; iteration++) {
            _bareRunSpMV( format, alg );
        }
        gpuTimer.stop();

        // wipe memory
        CUDA_CHECK( cudaFree(workspace) );
        return (gpuTimer.elapsed_msecs()/((float)repeatIters));
    }
    
    size_t getWorkspaceSize(SparseFormat format, DenseLayout layout, cusparseSpMMAlg_t alg, cusparseDnMatDescr_t &dnMatInputDescr,  cusparseDnMatDescr_t &dnMatOutputDescr)
    {
        size_t space;
        switch (layout) {
        case DENSE_COL_MAJOR:
            switch(format) {
            case SPARSE_FORMAT_COO:
                switch (alg) {
                case CUSPARSE_SPMM_ALG_DEFAULT:
                case CUSPARSE_SPMM_COO_ALG1:
                case CUSPARSE_SPMM_COO_ALG2:
                case CUSPARSE_SPMM_COO_ALG3:
                    goto bufferSizeNormal;
                default:
                    goto bufferSizeError;
                }
            case SPARSE_FORMAT_CSR:
                switch (alg) {
                    case CUSPARSE_SPMM_ALG_DEFAULT:
                    case CUSPARSE_SPMM_CSR_ALG1:
                        goto bufferSizeNormal;
                    default:
                        goto bufferSizeError;
                }
            default: goto bufferSizeError;
            }
        case DENSE_ROW_MAJOR:
            switch(format) {
            case SPARSE_FORMAT_COO:
                switch(alg) {
                case CUSPARSE_SPMM_ALG_DEFAULT:
                case CUSPARSE_SPMM_COO_ALG4:
                    goto bufferSizeNormal;
                default:
                    goto bufferSizeError;
                }
            case SPARSE_FORMAT_CSR:
                switch(alg) {
                case CUSPARSE_SPMM_ALG_DEFAULT:
                case CUSPARSE_SPMM_CSR_ALG2:
                    goto bufferSizeNormal;
                default:
                    goto bufferSizeError;
                }
            default: goto bufferSizeError;
            }
        default: goto bufferSizeError;
        }  
bufferSizeNormal:
        CUSPARSE_CHECK( cusparseSpMM_bufferSize(
                            handle, 
                            CUSPARSE_OPERATION_NON_TRANSPOSE, 
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha,
                            (format == SPARSE_FORMAT_CSR) ? csrDescr : cooDescr,
                            dnMatInputDescr,
                            &beta,
                            dnMatOutputDescr,
                            CUDA_R_32F,
                            alg,
                            &space
                ));
        return space;
bufferSizeError:
        std::cerr << "BufferSize error: Got layout " << layout << " format " << format << " but algorithm " << alg << std::endl;
        assert(0);
    }

    cusparseStatus_t _bareRunSpMM(SparseFormat format, cusparseDnMatDescr_t &dnMatInputDescr, cusparseDnMatDescr_t &dnMatOutputDescr, cusparseSpMMAlg_t alg)
    {
        return cusparseSpMM(
                    handle, 
                    CUSPARSE_OPERATION_NON_TRANSPOSE, // opA
                    CUSPARSE_OPERATION_NON_TRANSPOSE, // opB
                    &alpha, 
                    (format == SPARSE_FORMAT_CSR) ? csrDescr : cooDescr,
                    dnMatInputDescr, 
                    &beta,
                    dnMatOutputDescr,
                    CUDA_R_32F,
                    alg,
                    workspace);
    }

    void run(SparseFormat format, DenseLayout layout, int nv, cusparseSpMMAlg_t alg) {
        if (nv==1) {
            std::cout << "Warning, Got nv = 1, but SpMM algorithm.\n";
        }

        cusparseDnMatDescr_t dnMatInputDescr;
        cusparseDnMatDescr_t dnMatOutputDescr;

        CUSPARSE_CHECK( cusparseCreateDnMat(&dnMatInputDescr,
                            nc, // number of rows of input dense matrix B
                            nv, // number of columns of input dense matrix B
                            (layout == DENSE_COL_MAJOR) ? nc : nv,
                            dnInput,
                            CUDA_R_32F,
                            (layout == DENSE_COL_MAJOR) ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW
                            ));
        CUSPARSE_CHECK( cusparseCreateDnMat(&dnMatOutputDescr,
                            nr, // number of rows of input dense matrix C
                            nv, // number of columns of input dense matrix C
                            (layout == DENSE_COL_MAJOR) ? nr : nv,
                            dnOutput,
                            CUDA_R_32F,
                            (layout == DENSE_COL_MAJOR) ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW
                            ));
        
        // allocate workspace on device
        // this also provide gardient to algorithm layout matching. invalid combination will be throwed.
        size_t workspaceSize = getWorkspaceSize(format, layout, alg, dnMatInputDescr, dnMatOutputDescr);
        CUDA_CHECK( cudaMalloc(&workspace, workspaceSize ));

        // run spmv
        CUSPARSE_CHECK( _bareRunSpMM(format, dnMatInputDescr, dnMatOutputDescr, alg));

        // wipe memory
        CUDA_CHECK( cudaFree(workspace) );

    }
    
    float benchmark(SparseFormat format, DenseLayout layout, int nv, cusparseSpMMAlg_t alg, int repeatIters = 200, int warmupIters = 20)
    {
        if (nv==1) {
            std::cout << "Warning, Got nv = 1, but SpMM algorithm.\n";
        }
        
        
        cusparseDnMatDescr_t dnMatInputDescr;
        cusparseDnMatDescr_t dnMatOutputDescr;

        CUSPARSE_CHECK( cusparseCreateDnMat(&dnMatInputDescr,
                            nc, // number of rows of input dense matrix B
                            nv, // number of columns of input dense matrix B
                            (layout == DENSE_COL_MAJOR) ? nc : nv,
                            dnInput,
                            CUDA_R_32F,
                            (layout == DENSE_COL_MAJOR) ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW
                            ));
        CUSPARSE_CHECK( cusparseCreateDnMat(&dnMatOutputDescr,
                            nr, // number of rows of input dense matrix C
                            nv, // number of columns of input dense matrix C
                            (layout == DENSE_COL_MAJOR) ? nr : nv,
                            dnOutput,
                            CUDA_R_32F,
                            (layout == DENSE_COL_MAJOR) ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW
                            ));
        

        // allocate workspace on device
        // this also provide gardient to algorithm layout matching. invalid combination will be throwed.
        size_t workspaceSize = getWorkspaceSize(format, layout, alg, dnMatInputDescr, dnMatOutputDescr);
        CUDA_CHECK( cudaMalloc(&workspace, workspaceSize ));

        hgpu::GpuTimer gpuTimer;
        
        for (int iteration=0; iteration < warmupIters; iteration++) {
            _bareRunSpMM(format, dnMatInputDescr, dnMatOutputDescr, alg);
        }
        CUDA_CHECK( cudaPeekAtLastError());
        CUDA_CHECK( cudaDeviceSynchronize());

        gpuTimer.start();
        for (int iteration=0; iteration < repeatIters; iteration++) {
            _bareRunSpMM(format, dnMatInputDescr, dnMatOutputDescr, alg);
        }
        gpuTimer.stop();

        // wipe memory
        CUDA_CHECK( cudaFree(workspace) );
        return (gpuTimer.elapsed_msecs()/((float)repeatIters));
    }
    
};
#endif