/* Native FP64 integration blocks. No Python callbacks in the timestep loop.
 * q/p/g layout: Fortran (3,atoms,beads). Positive PES gradient, atomic units.
 * Reuses the existing cyclic Cayley kernel, without changing the physical map.
 * Each context owns all factors/workspaces. Every failed step is rolled back.
 */
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

typedef int32_t (*pes_fn)(int32_t,int32_t,const double*,double*,double*,
                         int32_t,int32_t*,int32_t*);
extern void drpmd_free_step(double*,double*,int,int,const double*,const double*,
                           double,double,const double*,const double*,const double*,
                           const double*,const double*,double*,int);

typedef struct {
    int32_t atoms, beads, nd, threads, pes_threads;
    size_t size;
    double dt, omega2, alpha, pair, sinverse[4];
    double *mass,*half,*spring,*inv,*lower,*z1,*z2,*work;
    double *old_q,*old_p,*old_v,*old_g;
    int32_t *status;
} engine;

/* detail = {bad_bead (zero based or -1), PES status, actual PES threads}.
 * Return codes: 0 success, 1 invalid argument, 2 allocation, 3 factorization,
 * 4 PES status/unwritten output, 5 nonfinite state/energy, 6 wrong team size.
 */
int32_t drpmd_engine_abi_version(void) { return 1; }

void drpmd_engine_destroy(void *ptr) {
    engine *e=ptr;
    if (!e) return;
    free(e->mass); free(e->half); free(e->spring); free(e->inv);
    free(e->lower); free(e->z1); free(e->z2); free(e->work);
    free(e->old_q); free(e->old_p); free(e->old_v); free(e->old_g);
    free(e->status); free(e);
}

static void solve(const engine *e,double *y) {
    const int b=e->beads;
    for (int k=1;k<b;++k) y[k]-=e->lower[k]*y[k-1];
    y[b-1]*=e->inv[b-1];
    for (int k=b-2;k>=0;--k) y[k]=(y[k]+e->alpha*y[k+1])*e->inv[k];
}

void *drpmd_engine_create(int32_t atoms,int32_t beads,const double *mass,
                          double beta,double dt,int32_t threads,int32_t pes_threads,int32_t *code) {
    *code=1;
    if (atoms<1 || atoms>INT_MAX/3 || beads<1 || threads<1 || pes_threads<1 || !mass ||
        !isfinite(beta) || beta<=0 || !isfinite(dt)) return NULL;
    size_t nd=(size_t)3*atoms;
    if ((size_t)beads>SIZE_MAX/sizeof(double)/nd) return NULL;
    for (int a=0;a<atoms;++a) if (!isfinite(mass[a]) || mass[a]<=0) return NULL;
    engine *e=calloc(1,sizeof(engine));
    *code=2;
    if (!e) return NULL;
    e->atoms=atoms; e->beads=beads; e->nd=(int32_t)nd;
    e->threads=threads; e->pes_threads=pes_threads; e->size=nd*beads; e->dt=dt;
    double omega=beads/beta;
    e->omega2=omega*omega;
    e->alpha=0.25*dt*dt*omega*omega;
    e->pair=1.0/(1.0+4.0*e->alpha);
    #define ALLOC(field,n) do { e->field=calloc((size_t)(n),sizeof(*e->field)); \
        if (!e->field) goto fail; } while(0)
    ALLOC(mass,atoms); ALLOC(half,nd); ALLOC(spring,nd);
    ALLOC(inv,beads); ALLOC(lower,beads); ALLOC(z1,beads); ALLOC(z2,beads);
    ALLOC(work,e->size); ALLOC(old_q,e->size); ALLOC(old_p,e->size);
    ALLOC(old_g,e->size); ALLOC(old_v,beads); ALLOC(status,beads);
    #undef ALLOC
    memcpy(e->mass,mass,(size_t)atoms*sizeof(double));
    *code=3;
    if (!isfinite(e->omega2) || !isfinite(4.0*e->alpha)) goto fail;
    for (int r=0;r<e->nd;++r) {
        e->half[r]=0.5*dt/mass[r/3];
        e->spring[r]=dt*mass[r/3]*omega*omega;
        if (!isfinite(e->half[r]) || !isfinite(e->spring[r])) goto fail;
    }
    e->sinverse[0]=e->sinverse[3]=1.0;
    if (beads>=3) {
        double diag=1.0+2.0*e->alpha;
        e->inv[0]=1.0/diag;
        for (int k=1;k<beads;++k) {
            e->lower[k]=-e->alpha*e->inv[k-1];
            e->inv[k]=1.0/(diag+e->lower[k]*e->alpha);
        }
        e->z1[0]=-e->alpha; solve(e,e->z1);
        e->z2[beads-1]=-e->alpha; solve(e,e->z2);
        double a=1.0+e->z1[beads-1], b=e->z2[beads-1];
        double c=e->z1[0], d=1.0+e->z2[0], det=a*d-b*c;
        if (det<=0 || !isfinite(det)) goto fail;
        e->sinverse[0]=d/det; e->sinverse[1]=-b/det;
        e->sinverse[2]=-c/det; e->sinverse[3]=a/det;
    }
    *code=0;
    return e;
fail:
    drpmd_engine_destroy(e);
    return NULL;
}

static int finite_array(const double *x,size_t n) {
    for (size_t i=0;i<n;++i) if (!isfinite(x[i])) return 0;
    return 1;
}

/* Deterministic scalar reductions: independent of thread count. */
static double energy(const engine *e,const double *q,const double *p,const double *v) {
    double kinetic=0,spring=0,potential=0;
    for (int b=0;b<e->beads;++b) {
        const int prev=b ? b-1 : e->beads-1;
        potential+=v[b];
        for (int r=0;r<e->nd;++r) {
            size_t i=(size_t)b*e->nd+r, j=(size_t)prev*e->nd+r;
            double m=e->mass[r/3], delta=q[i]-q[j];
            kinetic+=p[i]*p[i]/(2.0*m);
            spring+=0.5*m*e->omega2*delta*delta;
        }
    }
    return (kinetic+spring+potential)/e->beads;
}

/* targets are C [fragment][xyz]; group IDs partition all atoms into 0 or 1. */
static int project(const engine *e,double *x,const int32_t *group,
                    const double *targets,int positions) {
    double total[2]={0,0}, com[6]={0,0,0,0,0,0};
    for (int a=0;a<e->atoms;++a) {
        int f=group[a];
        if (f<0 || f>1) return 1;
        total[f]+=e->mass[a];
        for (int d=0;d<3;++d) {
            double sum=0;
            for (int b=0;b<e->beads;++b) sum+=x[(size_t)b*e->nd+3*a+d];
            com[3*f+d]+=(sum/e->beads)*(positions ? e->mass[a] : 1.0);
        }
    }
    if (total[0]<=0 || total[1]<=0 || !finite_array(total,2)) return 1;
    for (int f=0;f<2;++f) for (int d=0;d<3;++d) com[3*f+d]/=total[f];
    for (int b=0;b<e->beads;++b) for (int a=0;a<e->atoms;++a) for (int d=0;d<3;++d) {
        int k=3*group[a]+d;
        double shift=positions ? targets[k]-com[k] : -com[k]*e->mass[a];
        x[(size_t)b*e->nd+3*a+d]+=shift;
    }
    return 0;
}

int32_t drpmd_engine_project(void *ptr,double *p,const int32_t *group) {
    engine *e=ptr;
    if (!e || !p || !group) return 1;
    return project(e,p,group,NULL,0);
}

static int force(engine *e,pes_fn pes,const double *q,double *v,double *g,int32_t *detail) {
    detail[0]=-1; detail[1]=0; detail[2]=0;
    if (!finite_array(q,e->size)) return 5;
    for (int b=0;b<e->beads;++b) { v[b]=NAN; e->status[b]=-1; }
    for (size_t i=0;i<e->size;++i) g[i]=NAN;
    int32_t status=pes(e->atoms,e->beads,q,v,g,e->pes_threads,e->status,detail+2);
    detail[1]=status;
    for (int b=0;b<e->beads;++b) if (e->status[b]) {
        detail[0]=b; detail[1]=e->status[b]; return 4;
    }
    if (status || !finite_array(v,e->beads) || !finite_array(g,e->size)) return 4;
    if (detail[2]<1 || detail[2]>e->pes_threads) return 6;
    return 0;
}

int32_t drpmd_engine_start(void *ptr,pes_fn pes,const double *q,const double *p,
                           double *v,double *g,double *value,int32_t *detail) {
    engine *e=ptr;
    if (!e || !pes || !q || !p || !v || !g || !value || !detail) return 1;
    if (!finite_array(p,e->size)) return 5;
    int rc=force(e,pes,q,v,g,detail);
    if (rc) return rc;
    *value=energy(e,q,p,v);
    return isfinite(*value) ? 0 : 5;
}

int32_t drpmd_engine_advance(void *ptr,pes_fn pes,double *q,double *p,
    double *v,double *g,int32_t steps,const int32_t *group,const double *targets,
    const double *refresh,int32_t first_refresh,int32_t interval,int32_t nrefresh,
    double reference,double *value,double *drift,int32_t *done,int32_t *detail) {
    engine *e=ptr;
    if (!e || !pes || !q || !p || !v || !g || !done || !detail || !value || !drift) return 1;
    *done=0;
    if (steps<0 || !isfinite(reference) || !isfinite(*value) || !isfinite(*drift) || *drift<0) return 1;
    if (!finite_array(q,e->size) || !finite_array(p,e->size) ||
        !finite_array(g,e->size) || !finite_array(v,e->beads)) return 5;
    if (group && (!targets || !finite_array(targets,6))) return 1;
    if (group) {
        int counts[2]={0,0};
        for (int a=0;a<e->atoms;++a) {
            if (group[a]<0 || group[a]>1) return 1;
            counts[group[a]]++;
        }
        if (!counts[0] || !counts[1]) return 1;
    }
    if (nrefresh<0 || (nrefresh && (!group || !refresh || interval<1 || first_refresh<1))) return 1;
    if (nrefresh && ((int64_t)first_refresh+(int64_t)(nrefresh-1)*interval>steps ||
        (int64_t)first_refresh+(int64_t)nrefresh*interval<=steps ||
        !finite_array(refresh,(size_t)nrefresh*e->size))) return 1;
    int random_index=0;
    for (int s=0;s<steps;++s) {
        memcpy(e->old_q,q,e->size*sizeof(double)); memcpy(e->old_p,p,e->size*sizeof(double));
        memcpy(e->old_g,g,e->size*sizeof(double)); memcpy(e->old_v,v,(size_t)e->beads*sizeof(double));
        for (size_t i=0;i<e->size;++i) p[i]-=0.5*e->dt*g[i];
        if (group) project(e,p,group,NULL,0);
        drpmd_free_step(p,q,e->nd,e->beads,e->half,e->spring,e->alpha,e->pair,
                       e->inv,e->lower,e->z1,e->z2,e->sinverse,e->work,e->threads);
        if (group) project(e,q,group,targets,1);
        int rc=force(e,pes,q,v,g,detail);
        if (!rc) {
            for (size_t i=0;i<e->size;++i) p[i]-=0.5*e->dt*g[i];
            if (group) project(e,p,group,NULL,0);
            if (random_index<nrefresh && (int64_t)s+1==(int64_t)first_refresh+(int64_t)random_index*interval) {
                memcpy(p,refresh+(size_t)random_index*e->size,e->size*sizeof(double));
                ++random_index;
                project(e,p,group,NULL,0);
            }
            double new_value=energy(e,q,p,v);
            if (!finite_array(p,e->size) || !finite_array(q,e->size) || !isfinite(new_value)) rc=5;
            else { *value=new_value; *drift=fmax(*drift,fabs(new_value-reference)); }
        }
        if (rc) {
            memcpy(q,e->old_q,e->size*sizeof(double)); memcpy(p,e->old_p,e->size*sizeof(double));
            memcpy(g,e->old_g,e->size*sizeof(double)); memcpy(v,e->old_v,(size_t)e->beads*sizeof(double));
            return rc;
        }
        *done=s+1;
    }
    return 0;
}
