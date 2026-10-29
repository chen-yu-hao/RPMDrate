/* Analytic plumbing example, NOT a chemical PES. */
#include "drpmd_pes.h"
#include <math.h>
#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#endif
int32_t drpmd_pes_abi_version(void) { return 1; }
int32_t drpmd_pes_omp_enabled(void) {
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}
int32_t drpmd_pes_init(const char *path) { (void)path; return 0; }
int32_t drpmd_pes_eval(int32_t na, int32_t nb, const double *q, double *v,
                     double *g, int32_t nt, int32_t *status, int32_t *used) {
    if (na < 1 || nb < 1 || nt < 1) return 1;
    *used = 1;
    int error = 0;
    #pragma omp parallel num_threads(nt) reduction(|:error)
    {
        #ifdef _OPENMP
        #pragma omp single
        *used = omp_get_num_threads();
        #endif
        #pragma omp for schedule(static)
        for (int32_t b = 0; b < nb; ++b) {
            double e = 0.0;
            status[b] = 0;
            for (int32_t a = 0; a < na; ++a) {
                for (int32_t d = 0; d < 3; ++d) {
                    size_t i = ((size_t)b * na + a) * 3 + d;
                    g[i] = q[i];
                    e += 0.5*q[i]*q[i];
                    if (!isfinite(q[i]) || !isfinite(g[i])) status[b] = 2;
                }
            }
            v[b] = e;
            if (!isfinite(e)) status[b] = 2;
            error |= (status[b] != 0);
        }
    }
    return error;
}
