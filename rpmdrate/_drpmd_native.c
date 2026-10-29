/* Reentrant FP64 Cayley free-ring propagation for direct RPMD.
 *
 * This is the midpoint form used by rpmdrate/_main.f90.  All numerical
 * parameters, factors and scratch storage belong to the caller.  There is no
 * mutable global state.  OpenMP distributes complete Cartesian bead rings;
 * the dependent Thomas recurrence within each ring remains sequential.
 */
#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#endif

int drpmd_abi_version(void) { return 1; }
int drpmd_has_openmp(void) {
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}

void drpmd_kick(double *p, const double *gradient, size_t size,
                double interval, int threads) {
    #pragma omp parallel for schedule(static) num_threads(threads) if(threads > 1)
    for (size_t i = 0; i < size; ++i)
        p[i] -= interval * gradient[i];
}

void drpmd_free_step(double *p, double *q, int nd, int beads,
                     const double *half_dt_mass, const double *spring_mass,
                     double alpha, double pair_scale,
                     const double *inverse_diagonal, const double *lower,
                     const double *z1, const double *z2,
                     const double *sinverse, double *workspace, int threads) {
    /* p and q are Fortran-contiguous (nd, beads); each workspace row is
     * contiguous and owned by exactly one iteration of this parallel loop. */
    #pragma omp parallel for schedule(static) num_threads(threads) if(threads > 1)
    for (int row = 0; row < nd; ++row) {
        if (beads == 1) {
            q[row] += 2.0 * half_dt_mass[row] * p[row];
            continue;
        }
        if (beads == 2) {
            const size_t j0 = (size_t)row;
            const size_t j1 = (size_t)row + (size_t)nd;
            const double b1 = q[j0] + half_dt_mass[row] * p[j0];
            const double b2 = q[j1] + half_dt_mass[row] * p[j1];
            const double mid1 = ((1.0 + 2.0 * alpha) * b1 +
                                 2.0 * alpha * b2) * pair_scale;
            const double mid2 = (2.0 * alpha * b1 +
                                (1.0 + 2.0 * alpha) * b2) * pair_scale;
            const double impulse = 2.0 * spring_mass[row] * (mid1 - mid2);
            p[j0] -= impulse;
            p[j1] += impulse;
            q[j0] = 2.0 * mid1 - q[j0];
            q[j1] = 2.0 * mid2 - q[j1];
            continue;
        }

        double *mid = workspace + (size_t)row * (size_t)beads;
        for (int k = 0; k < beads; ++k) {
            const size_t index = (size_t)row + (size_t)nd * (size_t)k;
            mid[k] = q[index] + half_dt_mass[row] * p[index];
        }
        for (int k = 1; k < beads; ++k)
            mid[k] -= lower[k] * mid[k - 1];
        mid[beads - 1] *= inverse_diagonal[beads - 1];
        for (int k = beads - 2; k >= 0; --k)
            mid[k] = (mid[k] + alpha * mid[k + 1]) * inverse_diagonal[k];

        const double c1 = sinverse[0] * mid[beads - 1] + sinverse[1] * mid[0];
        const double c2 = sinverse[2] * mid[beads - 1] + sinverse[3] * mid[0];
        for (int k = 0; k < beads; ++k)
            mid[k] -= z1[k] * c1 + z2[k] * c2;

        for (int k = 0; k < beads; ++k) {
            const int previous = k ? k - 1 : beads - 1;
            const int next = k + 1 == beads ? 0 : k + 1;
            const size_t index = (size_t)row + (size_t)nd * (size_t)k;
            p[index] -= spring_mass[row] *
                       (2.0 * mid[k] - mid[previous] - mid[next]);
            q[index] = 2.0 * mid[k] - q[index];
        }
    }
}
