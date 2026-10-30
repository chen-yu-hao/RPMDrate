/* Default product check in C: centroid bond graph -> separated outgoing
 * fragments -> unordered composition match -> consecutive confirmations.
 * No Python callbacks, no allocation during checks, no writable global data.
 * Keep < bond cutoff, >= exit distance and strictly positive radial motion.
 */
#include "_drpmd_classifier.h"
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

typedef struct {
    int32_t atoms, beads, elements, channels, candidate, streak;
    int64_t checks;
    double exit_distance;
    int32_t *types, *nfragments, *formulas, *parent, *component, *counts, *used;
    double *mass, *cutoffs, *centroid, *momentum, *total, *centers, *velocity;
} classifier;

int32_t drpmd_classifier_abi_version(void) { return 1; }

void drpmd_classifier_destroy(void *ptr) {
    classifier *c=ptr;
    if (!c) return;
    free(c->types); free(c->nfragments); free(c->formulas); free(c->parent);
    free(c->component); free(c->counts); free(c->used); free(c->mass);
    free(c->cutoffs); free(c->centroid); free(c->momentum); free(c->total);
    free(c->centers); free(c->velocity); free(c);
}

void *drpmd_classifier_create(int32_t atoms,int32_t beads,int32_t elements,
    int32_t channels,const int32_t *types,const double *mass,const double *cutoffs,
    double exit_distance,const int32_t *nfragments,const int32_t *formulas,int32_t *status) {
    if (!status) return NULL;
    *status=1;
    if (atoms<2 || atoms>INT_MAX/3 || beads<1 || elements<1 || elements>atoms || channels<1 ||
        !types || !mass || !cutoffs || !nfragments || !formulas ||
        !isfinite(exit_distance) || exit_distance<=0) return NULL;
    size_t a=(size_t)atoms;
    if (a>SIZE_MAX/sizeof(int32_t)/(size_t)elements) return NULL;
    size_t ae=a*(size_t)elements;
    if (a>SIZE_MAX/sizeof(double)/a || ae>SIZE_MAX/sizeof(int32_t)/(size_t)channels ||
        (size_t)beads>SIZE_MAX/sizeof(double)/(3*a)) return NULL;
    for (int i=0;i<atoms;++i) {
        if (types[i]<0 || types[i]>=elements || !isfinite(mass[i]) || mass[i]<=0) return NULL;
        for (int j=0;j<atoms;++j) if (i!=j) {
            double r=cutoffs[(size_t)i*a+j];
            if (!isfinite(r) || r<=0 || r>=exit_distance || r!=cutoffs[(size_t)j*a+i]) return NULL;
        }
    }
    for (int k=0;k<channels;++k) {
        if (nfragments[k]<2 || nfragments[k]>atoms) return NULL;
        for (int f=0;f<atoms;++f) {
            int64_t sum=0;
            for (int t=0;t<elements;++t) {
                int v=formulas[((size_t)k*a+f)*elements+t];
                if (v<0 || v>atoms) return NULL;
                sum+=v;
            }
            if (f<nfragments[k] ? (sum<1 || sum>atoms) : sum!=0) return NULL;
        }
        for (int t=0;t<elements;++t) {
            int64_t sum=0; int actual=0;
            for (int f=0;f<nfragments[k];++f) sum+=formulas[((size_t)k*a+f)*elements+t];
            for (int i=0;i<atoms;++i) actual+=(types[i]==t);
            if (sum!=actual) return NULL;
        }
    }
    classifier *c=calloc(1,sizeof(*c));
    *status=2;
    if (!c) return NULL;
    c->atoms=atoms; c->beads=beads; c->elements=elements; c->channels=channels;
    c->exit_distance=exit_distance; c->candidate=-1;
    #define ALLOC(field,n) do { c->field=calloc((size_t)(n),sizeof(*c->field)); \
        if (!c->field) goto fail; } while(0)
    ALLOC(types,a); ALLOC(mass,a); ALLOC(cutoffs,a*a); ALLOC(nfragments,channels);
    ALLOC(formulas,(size_t)channels*ae); ALLOC(parent,a); ALLOC(component,a);
    ALLOC(counts,ae); ALLOC(used,a); ALLOC(centroid,3*a); ALLOC(momentum,3*a);
    ALLOC(total,a); ALLOC(centers,3*a); ALLOC(velocity,3*a);
    #undef ALLOC
    memcpy(c->types,types,a*sizeof(*types)); memcpy(c->mass,mass,a*sizeof(*mass));
    memcpy(c->cutoffs,cutoffs,a*a*sizeof(*cutoffs));
    memcpy(c->nfragments,nfragments,(size_t)channels*sizeof(*nfragments));
    memcpy(c->formulas,formulas,(size_t)channels*ae*sizeof(*formulas));
    *status=0;
    return c;
fail:
    drpmd_classifier_destroy(c);
    return NULL;
}

static int root(classifier *c,int i) {
    while (c->parent[i]!=i) {
        c->parent[i]=c->parent[c->parent[i]];
        i=c->parent[i];
    }
    return i;
}

int32_t drpmd_classifier_dimensions(void *ptr,int32_t atoms,int32_t beads) {
    classifier *c=ptr;
    return c && c->atoms==atoms && c->beads==beads;
}

int64_t drpmd_classifier_checks(void *ptr) {
    classifier *c=ptr;
    return c ? c->checks : -1;
}

int32_t drpmd_classifier_evaluate(void *ptr,const double *q,const double *p,int32_t *label) {
    classifier *c=ptr;
    if (!c || !q || !p || !label) return 1;
    *label=-1;
    int n=c->atoms, types=c->elements;
    size_t nd=(size_t)3*n;
    for (int i=0;i<n;++i) {
        c->parent[i]=i; c->component[i]=-1;
        for (int d=0;d<3;++d) {
            double x=0, v=0;
            for (int b=0;b<c->beads;++b) {
                size_t j=(size_t)b*nd+3*(size_t)i+d;
                x+=q[j]; v+=p[j];
            }
            x/=c->beads; v/=c->beads;
            if (!isfinite(x) || !isfinite(v)) return 3;
            c->centroid[3*i+d]=x; c->momentum[3*i+d]=v;
        }
    }
    for (int i=0;i<n;++i) for (int j=i+1;j<n;++j) {
        double norm2=0;
        for (int d=0;d<3;++d) {
            double delta=c->centroid[3*i+d]-c->centroid[3*j+d];
            norm2+=delta*delta;
        }
        if (!isfinite(norm2)) return 3;
        if (sqrt(norm2)<c->cutoffs[(size_t)i*n+j]) {
            int a=root(c,i), b=root(c,j);
            if (a!=b) c->parent[b]=a;
        }
    }
    int parts=0;
    for (int i=0;i<n;++i) {
        int r=root(c,i);
        if (c->component[r]<0) c->component[r]=parts++;
    }
    if (parts<2) return 0;
    memset(c->counts,0,(size_t)n*types*sizeof(*c->counts));
    memset(c->total,0,(size_t)n*sizeof(double));
    memset(c->centers,0,nd*sizeof(double)); memset(c->velocity,0,nd*sizeof(double));
    for (int i=0;i<n;++i) {
        int f=c->component[root(c,i)];
        ++c->counts[(size_t)f*types+c->types[i]];
        c->total[f]+=c->mass[i];
        for (int d=0;d<3;++d) {
            c->centers[3*f+d]+=c->mass[i]*c->centroid[3*i+d];
            c->velocity[3*f+d]+=c->momentum[3*i+d];
        }
    }
    for (int f=0;f<parts;++f) {
        if (!isfinite(c->total[f])) return 3;
        for (int d=0;d<3;++d) {
            c->centers[3*f+d]/=c->total[f]; c->velocity[3*f+d]/=c->total[f];
            if (!isfinite(c->centers[3*f+d]) || !isfinite(c->velocity[3*f+d])) return 3;
        }
    }
    for (int i=0;i<parts;++i) for (int j=i+1;j<parts;++j) {
        double norm2=0, dot=0;
        for (int d=0;d<3;++d) {
            double r=c->centers[3*j+d]-c->centers[3*i+d];
            norm2+=r*r;
            dot+=r*(c->velocity[3*j+d]-c->velocity[3*i+d]);
        }
        if (!isfinite(norm2) || !isfinite(dot)) return 3;
        if (sqrt(norm2)<c->exit_distance || dot<=0) return 0;
    }
    for (int k=0;k<c->channels;++k) {
        if (c->nfragments[k]!=parts) continue;
        memset(c->used,0,(size_t)parts*sizeof(*c->used));
        int matched=0;
        for (int f=0;f<parts;++f) {
            for (int j=0;j<parts;++j) {
                if (!c->used[j] && !memcmp(c->counts+(size_t)f*types,
                    c->formulas+((size_t)k*n+j)*types,(size_t)types*sizeof(*c->counts))) {
                    c->used[j]=1; ++matched; break;
                }
            }
        }
        if (matched==parts) { *label=k; break; }
    }
    return 0;
}

int32_t drpmd_classifier_observe(void *ptr,const double *q,const double *p,
                               int32_t persistence,int32_t *resolved) {
    classifier *c=ptr;
    if (!c || persistence<1 || !resolved || c->checks==INT64_MAX) return 1;
    *resolved=-1;
    int32_t label;
    int rc=drpmd_classifier_evaluate(c,q,p,&label);
    if (rc) return rc;
    ++c->checks;
    c->streak=label<0 ? 0 : (label==c->candidate ? (c->streak<INT_MAX ? c->streak+1 : INT_MAX) : 1);
    c->candidate=label;
    if (label>=0 && c->streak>=persistence) *resolved=label;
    return 0;
}
