/* Host dispatch only: use the caller's existing cuBLAS handle/stream/workspace.
 * ABI argument types match cublasGemmEx/GetMathMode/SetMathMode in CUDA 13.
 * No device code, atomics, padded GEMMs, algorithm search, or persistent handle.
 */
#include <stdint.h>
#include <stddef.h>

typedef int (*gemm_fn)(void *, int, int, int, int, int, const void *,
                       const void *, int, int, const void *, int, int,
                       const void *, void *, int, int, int, int);
typedef int (*get_mode_fn)(void *, int *);
typedef int (*set_mode_fn)(void *, int);

int decisionvla_cublas_forward(void *gemm_ptr, void *get_ptr, void *set_ptr,
    void *handle, const uint16_t *x, const uint16_t *w, uint16_t *y,
    const int32_t *ends, int groups, int rows, int n, int k, int math_mode) {
    gemm_fn gemm = (gemm_fn)gemm_ptr;
    get_mode_fn get_mode = (get_mode_fn)get_ptr;
    set_mode_fn set_mode = (set_mode_fn)set_ptr;
    if (groups <= 0 || rows < 0 || n <= 0 || k <= 0) return -1;
    int start = 0;
    for (int e = 0; e < groups; ++e) {
        if (ends[e] < start || ends[e] > rows) return -2;
        start = ends[e];
    }
    if (start != rows) return -3;
    int previous = 0;
    int status = get_mode(handle, &previous);
    if (status) return status;
    status = set_mode(handle, math_mode);
    if (status) return status;
    const float alpha = 1.0f, beta = 0.0f;
    start = 0;
    for (int e = 0; e < groups; ++e) {
        int count = ends[e] - start;
        if (count > 0) {
            /* Column-major Y.T = W X.T, same per-expert dimensions as torch.mm.
             * CUDA_R_16BF=14, CUBLAS_COMPUTE_32F=68, DEFAULT_TENSOR_OP=99.
             */
            status = gemm(handle, 1, 0, n, count, k, &alpha,
                w + (size_t)e*n*k, 14, k, x + (size_t)start*k, 14, k,
                &beta, y + (size_t)start*n, 14, n, 68, 99);
            if (status) break;
        }
        start = ends[e];
    }
    int restored = set_mode(handle, previous);
    return status ? status : restored;
}
