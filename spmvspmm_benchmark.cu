#include "spmvspmm.h"
#include "cusparse_helper.h"
#include "util/spmatrix.hpp"

const int _MAXNV=128;
const int _MINNV=1;

using IdxArray=std::vector<int>;
using ValArray=std::vector<float>;

void cpuSpmvReference(DenseLayout layout, int nr, int nc, int nnz, int nv, IdxArray& rowPtr, IdxArray& colIdx, ValArray& values, ValArray& input, ValArray& outputRef)
{
    if (layout==DENSE_COL_MAJOR)
    {
        for (int k=0; k<nv; k++) {
            float *vec = input.data() + k*nc;
            for (int r=0; r<nr; r++) {
                float val = 0;
                for (int p=rowPtr[r]; p<rowPtr[r+1]; p++) {
                    val += values[p] * vec[colIdx[p]];
                }
                outputRef[k*nr + r] = val;
            }
        }
    }
    else { // RowMajor
        for (int r=0; r<nr; r++) {
            ValArray vals(nv, 0.0);
            for (int p=rowPtr[r]; p<rowPtr[r+1]; p++) {
                for (int k=0; k<nv; k++) {
                    vals[k] += values[p] * 
                                input[ colIdx[p] * nv + k];
                }
            }
            for (int k=0; k<nv; k++) 
                outputRef[ r*nv + k ] = vals[k];
        }
    }
}

void generateCsrRowPtrArray(int nr, IdxArray& rowPtrBuf, IdxArray& rowIdxBuf)
{
    rowPtrBuf = IdxArray( nr+1, 0);
    int curr_row = rowIdxBuf[0];
    int curr_pos = 0;
    int nnz = rowIdxBuf.size();
    auto it = rowPtrBuf.begin() + curr_row; // no need initialization because already 0
    int next_row;
    // for every row
    //    for (; curr_row < nr; )
    //    int p1 = 0; int p2 = 0;
    //    int curr_offset = p1; int curr_row = rowIdxBuf[p1]; int next_row;
    //    while (p2 < nr)
      //      {
	//	while (p2 <= curr_row)
	  //	  rowPtrBuf[p2++] = curr_offset;
	//	while ( p1 < nnz-1 && (next_row = rowIdxBuf[++p1])==curr_row);
	//	if (p1==nnz) break;
	//	curr_offset = p1;
	//	curr_row = next_row;
	//  }
    
    for (; it != rowPtrBuf.end(); ) {
        do {
            ++curr_pos;
        } while (curr_pos < nnz && ((next_row = rowIdxBuf[curr_pos]) == curr_row));
        do {
            *(++it) = curr_pos;
            curr_row++;
        } while (curr_row < nr && curr_row < next_row);
        if (curr_pos == nnz) break;
    }
    while (++it < rowPtrBuf.end()) *it = curr_pos;

    std::cerr << "Convert format finished. last offset is " << rowPtrBuf[nr] << std::endl;
}

template<typename DType>
int calcMaxNv(int nr, int nc, int nnz, int devId=0)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, devId);
    size_t deviceMemSize = prop.totalGlobalMem;
    fprintf(stderr, "Global Mem: %ld\n",deviceMemSize);
    size_t spMatSize = (nr+1 + nnz + nnz)*sizeof(int) + nnz*sizeof(DType);
    long maxNv = (deviceMemSize - spMatSize) / sizeof(DType) / ((nr+nc));
    int nv=_MAXNV;
    while (nv>maxNv && nv>0) nv /=2;
    return nv;
}

// utility function to check two results match or not
template<DenseLayout host_layout, DenseLayout dev_layout = host_layout, bool throwOnlyFirstError = true>
bool checkResult(int nrows, int ncols, float *ref, float *out, float eps = 1e-3)
{
    bool passed = true;
    for (int i=0; i<nrows; i++)
    {
        for (int j=0; j<ncols; j++) 
        {
            int idx1 = (host_layout == DENSE_ROW_MAJOR) ? (i * ncols + j) : (j * nrows + i);
            int idx2 = (dev_layout == DENSE_ROW_MAJOR) ? (i * ncols + j) : (j * nrows + i);
            if (fabs( out[idx2] - ref[idx1]) > eps)
            {
                passed = false;
                fprintf(stderr, "Check Failed out[%d,%d] %f != %f\n", i, j, out[idx2], ref[idx1]);
                if (throwOnlyFirstError) 
                    goto checkResultReturn;
                break;
            }
        }
    }
    // if (passed) fprintf(stderr, "Check passed;\n");
checkResultReturn:
    return passed;
}

bool checkAlgorithms(
    int nr, // number of rows of sparse matrix
    int nc, // number of columns of sparse matrix
    int nnz, // number of non-zeros in sparse matrix
    int maxNv, // maximum number of columns in dense matrix
    IdxArray& rowPtr, 
    IdxArray& colIdx, 
    ValArray& values, 
    ValArray& input, 
    ValArray& outputRef,
#if ( CUDA_VERSION >= 11000 )
    CusparseSpmmProblem cusparseProblem,
#else 
    CusparseCsrmmProblem cusparseProblem,
#endif
    SpmvSpmmProblem spmvspmmProblem,
    hgpu::CUArray<float>& output
)
{
    bool passed = true;

#if ( CUDA_VERSION >= 11000 )
#define CUSPARSE_CHECK_SWITCH(format, layout, nv, alg)  \
CUDA_CHECK(cudaMemset(output.get_device_data(), 0, (nr*nv*sizeof(float)))); \
cusparseProblem.run(format, layout, nv, alg);           \
output.sync_host();                                     \
passed = passed && checkResult<layout>(nr, nv, outputRef.data(), output.get_host_data());
#else
#define CUSPARSE_CHECK_SWITCH(layout, nv)               \
CUDA_CHECK(cudaMemset(output.get_device_data(), 0, (nr*nv*sizeof(float)))); \
cusparseProblem.run(layout, nv);                        \
output.sync_host();                                     \
passed = passed && checkResult<layout>(nr, nv, outputRef.data(), output.get_host_data()) ;
#endif

#define SPMV_SPMM_CHECK_SWITCH(layout, nv, alg)         \
CUDA_CHECK(cudaMemset(output.get_device_data(), 0, (nr*nv*sizeof(float)))); \
spmvspmmProblem.run(layout, nv, alg);                   \
output.sync_host();                                     \
passed = passed && checkResult<layout>(nr, nv, outputRef.data(), output.get_host_data()) ;


    // row major
    // for (int layout = DENSE_ROW_MAJOR; layout != DenseLayout_End; ++layout)

    for (int nv=_MINNV; nv<= maxNv; nv *=2)
    {
        // fprintf(stderr, "nv=%d\n", nv);
        //
        // row
        //
        cpuSpmvReference(DENSE_ROW_MAJOR, nr, nc, nnz, nv, rowPtr, colIdx, values, input, outputRef);

        // test cusparse
#if ( CUDA_VERSION >= 11000 )
        if (nv==1)
        {
            CUSPARSE_CHECK_SWITCH(SPARSE_FORMAT_CSR, DENSE_ROW_MAJOR, nv, CUSPARSE_MV_ALG_DEFAULT)
            CUSPARSE_CHECK_SWITCH(SPARSE_FORMAT_CSR, DENSE_ROW_MAJOR, nv, CUSPARSE_CSRMV_ALG1)
            CUSPARSE_CHECK_SWITCH(SPARSE_FORMAT_COO, DENSE_ROW_MAJOR, nv, CUSPARSE_MV_ALG_DEFAULT)
            CUSPARSE_CHECK_SWITCH(SPARSE_FORMAT_COO, DENSE_ROW_MAJOR, nv, CUSPARSE_COOMV_ALG)
        }
        else {
            CUSPARSE_CHECK_SWITCH(SPARSE_FORMAT_CSR, DENSE_ROW_MAJOR, nv, CUSPARSE_SPMM_ALG_DEFAULT)
            CUSPARSE_CHECK_SWITCH(SPARSE_FORMAT_CSR, DENSE_ROW_MAJOR, nv, CUSPARSE_SPMM_CSR_ALG2)
            CUSPARSE_CHECK_SWITCH(SPARSE_FORMAT_COO, DENSE_ROW_MAJOR, nv, CUSPARSE_SPMM_ALG_DEFAULT)
            CUSPARSE_CHECK_SWITCH(SPARSE_FORMAT_COO, DENSE_ROW_MAJOR, nv, CUSPARSE_SPMM_COO_ALG4)              
        }
#else
        CUDA_CHECK(cudaMemset(output.get_device_data(), 0, (nr*nv*sizeof(float)))); \
        cusparseProblem.run(DENSE_ROW_MAJOR, nv);                        \
        output.sync_host();
        passed = passed && checkResult<DENSE_ROW_MAJOR, DENSE_COL_MAJOR>(nr, nv, outputRef.data(), output.get_host_data()) ;
#endif
        // test cudaSpmm

        SPMV_SPMM_CHECK_SWITCH(DENSE_ROW_MAJOR, nv, ALG_CSR_SCALAR)
        SPMV_SPMM_CHECK_SWITCH(DENSE_ROW_MAJOR, nv, ALG_CSR_VECTOR)
        SPMV_SPMM_CHECK_SWITCH(DENSE_ROW_MAJOR, nv, ALG_COO_SCALAR)
        SPMV_SPMM_CHECK_SWITCH(DENSE_ROW_MAJOR, nv, ALG_COO_VECTOR)

        //
        // col
        //
        cpuSpmvReference(DENSE_COL_MAJOR, nr, nc, nnz, nv, rowPtr, colIdx, values, input, outputRef);

        // test cusparse
#if ( CUDA_VERSION >= 11000 )
        if (nv==1)
        {
            CUSPARSE_CHECK_SWITCH(SPARSE_FORMAT_CSR, DENSE_COL_MAJOR, nv, CUSPARSE_MV_ALG_DEFAULT)
            CUSPARSE_CHECK_SWITCH(SPARSE_FORMAT_CSR, DENSE_COL_MAJOR, nv, CUSPARSE_CSRMV_ALG1)
            CUSPARSE_CHECK_SWITCH(SPARSE_FORMAT_COO, DENSE_COL_MAJOR, nv, CUSPARSE_MV_ALG_DEFAULT)
            CUSPARSE_CHECK_SWITCH(SPARSE_FORMAT_COO, DENSE_COL_MAJOR, nv, CUSPARSE_COOMV_ALG)
        }
        else {
            CUSPARSE_CHECK_SWITCH(SPARSE_FORMAT_CSR, DENSE_COL_MAJOR, nv, CUSPARSE_SPMM_ALG_DEFAULT)
            CUSPARSE_CHECK_SWITCH(SPARSE_FORMAT_CSR, DENSE_COL_MAJOR, nv, CUSPARSE_SPMM_CSR_ALG1)
            CUSPARSE_CHECK_SWITCH(SPARSE_FORMAT_COO, DENSE_COL_MAJOR, nv, CUSPARSE_SPMM_ALG_DEFAULT)
            CUSPARSE_CHECK_SWITCH(SPARSE_FORMAT_COO, DENSE_COL_MAJOR, nv, CUSPARSE_SPMM_COO_ALG1)
            CUSPARSE_CHECK_SWITCH(SPARSE_FORMAT_COO, DENSE_COL_MAJOR, nv, CUSPARSE_SPMM_COO_ALG2)
            CUSPARSE_CHECK_SWITCH(SPARSE_FORMAT_COO, DENSE_COL_MAJOR, nv, CUSPARSE_SPMM_COO_ALG3)
        }
#else
        CUSPARSE_CHECK_SWITCH(DENSE_COL_MAJOR, nv)
#endif
        // test cudaSpmm

        SPMV_SPMM_CHECK_SWITCH(DENSE_COL_MAJOR, nv, ALG_CSR_SCALAR)
        SPMV_SPMM_CHECK_SWITCH(DENSE_COL_MAJOR, nv, ALG_CSR_VECTOR)
        SPMV_SPMM_CHECK_SWITCH(DENSE_COL_MAJOR, nv, ALG_COO_SCALAR)
        SPMV_SPMM_CHECK_SWITCH(DENSE_COL_MAJOR, nv, ALG_COO_VECTOR)
    }

    return passed;
}


void benchmarkAlgorithms(
    int nr, // number of rows of sparse matrix
    int nc, // number of columns of sparse matrix
    int nnz, // number of non-zeros in sparse matrix
    int maxNv, // maximum number of columns in dense matrix
    CusparseSpmmProblem cusparseProblem,
    SpmvSpmmProblem spmvspmmProblem,
    std::string filename,
    bool profile = false
)
{

    float flop, runtime;
    
#if ( CUDA_VERSION >= 11000 )
#define CUSPARSE_BENCHMARK_SWITCH(format, layout, nv, alg)      \
runtime = cusparseProblem.benchmark(format, layout, nv, alg);   \
std::cout << "," << (flop/runtime);                         
#else
#define CUSPARSE_BENCHMARK_SWITCH(layout, nv)                   \
runtime = cusparseProblem.benchmark(layout, nv);                \
std::cout << "," << (flop/runtime);
#endif

#define SPMV_SPMM_BENCHMARK_SWITCH(layout, nv, alg)             \
runtime = spmvspmmProblem.benchmark(layout, nv, alg);           \
std::cout << "," << (flop/runtime);       

    if (profile)
    {
        for (int nv = _MINNV; nv <= maxNv; nv *= 2)
        {
        spmvspmmProblem.dry_run(DENSE_ROW_MAJOR, nv, ALG_CSR_SCALAR, 5);
        spmvspmmProblem.dry_run(DENSE_ROW_MAJOR, nv, ALG_CSR_VECTOR, 5);
        spmvspmmProblem.dry_run(DENSE_ROW_MAJOR, nv, ALG_COO_SCALAR, 5);
        spmvspmmProblem.dry_run(DENSE_ROW_MAJOR, nv, ALG_COO_VECTOR, 5);
        spmvspmmProblem.dry_run(DENSE_COL_MAJOR, nv, ALG_CSR_SCALAR, 5);
        spmvspmmProblem.dry_run(DENSE_COL_MAJOR, nv, ALG_CSR_VECTOR, 5);
        spmvspmmProblem.dry_run(DENSE_COL_MAJOR, nv, ALG_COO_SCALAR, 5);
        spmvspmmProblem.dry_run(DENSE_COL_MAJOR, nv, ALG_COO_VECTOR, 5);
        }
        return;
    }
    
    // std::cout << "CSV Result\n";
    // std::cout << "filename,KDense,csr_scalar_row,csr_vector_row,coo_scalar_row,coo_vector_row,csr_scalar_col,csr_vector_col,coo_scalar_col,coo_vector_col,cusparse_csr_row_default,cusparse_csr_row_alg,cusparse_coo_row_default,cusparse_coo_row_alg,cusparse_csr_col_default,cusparse_csr_col_alg,cusparse_coo_col_default,cusparse_coo_col_alg1,cusparse_coo_col_alg2,cusparse_coo_col_alg3,\n";

    for (int nv = _MINNV; nv <= maxNv; nv *= 2)
    {
        flop = (float)nnz / 1e6 * 2 * nv;
        std::cout << filename << "," << nv;

        //
        // run cudaSpmm
        //
        SPMV_SPMM_BENCHMARK_SWITCH(DENSE_ROW_MAJOR, nv, ALG_CSR_SCALAR)
        SPMV_SPMM_BENCHMARK_SWITCH(DENSE_ROW_MAJOR, nv, ALG_CSR_VECTOR)
        SPMV_SPMM_BENCHMARK_SWITCH(DENSE_ROW_MAJOR, nv, ALG_COO_SCALAR)
        SPMV_SPMM_BENCHMARK_SWITCH(DENSE_ROW_MAJOR, nv, ALG_COO_VECTOR)
        SPMV_SPMM_BENCHMARK_SWITCH(DENSE_COL_MAJOR, nv, ALG_CSR_SCALAR)
        SPMV_SPMM_BENCHMARK_SWITCH(DENSE_COL_MAJOR, nv, ALG_CSR_VECTOR)
        SPMV_SPMM_BENCHMARK_SWITCH(DENSE_COL_MAJOR, nv, ALG_COO_SCALAR)
        SPMV_SPMM_BENCHMARK_SWITCH(DENSE_COL_MAJOR, nv, ALG_COO_VECTOR)

        //
        // run cusparse
        //
        if (nv==1)
        {
            CUSPARSE_BENCHMARK_SWITCH(SPARSE_FORMAT_CSR, DENSE_ROW_MAJOR, nv, CUSPARSE_MV_ALG_DEFAULT)
            CUSPARSE_BENCHMARK_SWITCH(SPARSE_FORMAT_CSR, DENSE_ROW_MAJOR, nv, CUSPARSE_CSRMV_ALG1)
            CUSPARSE_BENCHMARK_SWITCH(SPARSE_FORMAT_COO, DENSE_ROW_MAJOR, nv, CUSPARSE_MV_ALG_DEFAULT)
            CUSPARSE_BENCHMARK_SWITCH(SPARSE_FORMAT_COO, DENSE_ROW_MAJOR, nv, CUSPARSE_COOMV_ALG)
        }
        else 
        {
            CUSPARSE_BENCHMARK_SWITCH(SPARSE_FORMAT_CSR, DENSE_ROW_MAJOR, nv, CUSPARSE_SPMM_ALG_DEFAULT)
            CUSPARSE_BENCHMARK_SWITCH(SPARSE_FORMAT_CSR, DENSE_ROW_MAJOR, nv, CUSPARSE_SPMM_CSR_ALG2)
            CUSPARSE_BENCHMARK_SWITCH(SPARSE_FORMAT_COO, DENSE_ROW_MAJOR, nv, CUSPARSE_SPMM_ALG_DEFAULT)
            CUSPARSE_BENCHMARK_SWITCH(SPARSE_FORMAT_COO, DENSE_ROW_MAJOR, nv, CUSPARSE_SPMM_COO_ALG4)

            CUSPARSE_BENCHMARK_SWITCH(SPARSE_FORMAT_CSR, DENSE_COL_MAJOR, nv, CUSPARSE_SPMM_ALG_DEFAULT)
            CUSPARSE_BENCHMARK_SWITCH(SPARSE_FORMAT_CSR, DENSE_COL_MAJOR, nv, CUSPARSE_SPMM_CSR_ALG1)
            CUSPARSE_BENCHMARK_SWITCH(SPARSE_FORMAT_COO, DENSE_COL_MAJOR, nv, CUSPARSE_SPMM_ALG_DEFAULT)
            CUSPARSE_BENCHMARK_SWITCH(SPARSE_FORMAT_COO, DENSE_COL_MAJOR, nv, CUSPARSE_SPMM_COO_ALG1)
            CUSPARSE_BENCHMARK_SWITCH(SPARSE_FORMAT_COO, DENSE_COL_MAJOR, nv, CUSPARSE_SPMM_COO_ALG2)
            CUSPARSE_BENCHMARK_SWITCH(SPARSE_FORMAT_COO, DENSE_COL_MAJOR, nv, CUSPARSE_SPMM_COO_ALG3)
        } 
        
        std::cout << std::endl;
    }
}

int main(int argc, char** argv)
{
    if (argc<2) {
        std::cerr << "usage: ./exec <filename> " << std::endl;
        exit( EXIT_SUCCESS );
    }
    
    bool profile = false;
    if (argc>2) {
        std::string profile_str="profile";
        if (argv[2] == profile_str) {
            profile = true;
        }
    }

    int nr, nc, nnz, maxNv;
    IdxArray rowIdxBuf, colIdxBuf, rowPtrBuf;
    ValArray valuesBuf;
    
    // load sparse matrix meta data from the file
    readMtx<float>(argv[1], rowIdxBuf, colIdxBuf, valuesBuf, nr, nc, nnz);
         
    // compute the maximum dense matrix size

    maxNv = calcMaxNv<float>(nr, nc, nnz);
    if (profile) 
    {
        maxNv = atoi(argv[3]);
    }

    fprintf(stderr, "Read file finished. nrows = %d, ncols = %d, nnz = %d, maxNv = %d\n", nr, nc, nnz, maxNv);
    if (maxNv==0) {
        std::cerr << "Error: get maxNv=0. " << std::endl;
        exit( EXIT_FAILURE );
    }

    // generate coo and csr format
    // the meta data is already sorted by row - col indices
    generateCsrRowPtrArray(nr, rowPtrBuf, rowIdxBuf);

    hgpu::CUArray<int> rowPtrConst(rowPtrBuf);
    hgpu::CUArray<int> rowIdxConst(rowIdxBuf);
    hgpu::CUArray<int> colIdxConst(colIdxBuf);
    hgpu::CUArray<float> valuesConst(valuesBuf);
    rowPtrConst.sync_device();
    rowIdxConst.sync_device();
    colIdxConst.sync_device();
    valuesConst.sync_device();

    // prepare dense matrix
    hgpu::CUArray<float> denseInput, denseOutput;
    denseInput.init_random( nc * maxNv );
    denseOutput.init_zeros( nr * maxNv );
    denseInput.sync_device();
    denseOutput.sync_device();
    ValArray denseOutputRef( nr * maxNv, 0.0);

    // define applications
    static_assert( (CUDA_VERSION > 0), "CUDA_VERSION not defined");
    CusparseSpmmProblem cusparseProblem(
        nr, nc, nnz, maxNv,
        rowPtrConst.get_device_data(),
        rowIdxConst.get_device_data(),
        colIdxConst.get_device_data(),
        valuesConst.get_device_data(),
        denseInput.get_device_data(),
        denseOutput.get_device_data()
    );

    SpmvSpmmProblem spmvspmmProblem(
        nr, nc, nnz, maxNv,
        rowPtrConst.get_device_data(),
        rowIdxConst.get_device_data(),
        colIdxConst.get_device_data(),
        valuesConst.get_device_data(),
        denseInput.get_device_data(),
        denseOutput.get_device_data()
    );

    bool passed = true;
    // Run test on all algorithms
    passed = checkAlgorithms(nr, nc, nnz, maxNv, rowPtrBuf, colIdxBuf, valuesBuf, denseInput.get_host_array(), denseOutputRef, cusparseProblem, spmvspmmProblem, denseOutput);
    // Run benchmark on all algorithms
    if (passed) {
        std::string filename = std::string(argv[1]);
        benchmarkAlgorithms(nr, nc, nnz, maxNv, 
        cusparseProblem, 
        spmvspmmProblem, filename, profile);
    }
    return (passed ? 0 : -1);
}
