#ifndef DRPMD_PES_H
#define DRPMD_PES_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Return 1. All functions use the C calling convention. */
int32_t drpmd_pes_abi_version(void);
/* Return 1 only when OpenMP support is compiled in. */
int32_t drpmd_pes_omp_enabled(void);
/* Called serially before evaluation; 0 success, nonzero failure.
 * UTF-8 NUL-terminated model path, empty when unused. Freeze model data here.
 * Do not STOP/abort/exit on errors. A process may load only one model per DSO.
 */
int32_t drpmd_pes_init(const char *model_path);
/* q and gradient are Fortran (3,natoms,nbeads), i.e. C [bead][atom][xyz].
 * q is read-only, FP64, bohr. energy: [nbeads], Hartree.
 * gradient is POSITIVE dV/dq (Hartree/bohr), NOT force.
 * No reductions across beads; each bead is an independent PES call.
 * status: [nbeads], one status per bead (0=success).
 * used_threads: actual OpenMP team size (>=1, <=nthreads).
 * On success fill EVERY output element and return 0. On failure return nonzero.
 * All evaluation work arrays must be thread-local; initialized weights read-only.
 * Never retain caller-owned pointers. No Python callbacks in the OpenMP region.
 */
int32_t drpmd_pes_eval(int32_t natoms, int32_t nbeads, const double *q,
                     double *energy, double *gradient, int32_t nthreads,
                     int32_t *status, int32_t *used_threads);
#ifdef __cplusplus
}
#endif
#endif
