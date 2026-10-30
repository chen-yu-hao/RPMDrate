#ifndef DRPMD_CLASSIFIER_H
#define DRPMD_CLASSIFIER_H
#include <stdint.h>
/* Private reentrant centroid-connectivity classifier. All input tables are
 * copied at construction. q/p: Fortran (3,atoms,beads), atomic units.
 * Label -1 = not resolved, 0 = reactants, 1..channels-1 = configured product.
 * Formula tables are C [channel][fragment slot up to atoms][element].
 * This encodes composition only, not electronic state or atom identity.
 */
int32_t drpmd_classifier_abi_version(void);
void *drpmd_classifier_create(int32_t atoms, int32_t beads, int32_t elements,
    int32_t channels, const int32_t *types, const double *mass,
    const double *cutoffs, double exit_distance, const int32_t *nfragments,
    const int32_t *formulas, int32_t *status);
void drpmd_classifier_destroy(void *ptr);
int32_t drpmd_classifier_evaluate(void *ptr, const double *q, const double *p,
                                int32_t *label);
int32_t drpmd_classifier_observe(void *ptr, const double *q, const double *p,
                               int32_t persistence, int32_t *resolved);
int32_t drpmd_classifier_dimensions(void *ptr, int32_t atoms, int32_t beads);
int64_t drpmd_classifier_checks(void *ptr);
#endif
