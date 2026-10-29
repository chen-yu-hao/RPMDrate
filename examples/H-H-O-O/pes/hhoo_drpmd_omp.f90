! C ABI adapter only. Link the UNCHANGED hhoo_1014_analytic.f90 first.
! Model initialization is serial; evaluations only read frozen model arrays.
! Compile with -fPIC -fopenmp -frecursive -O3, never -ffast-math.
module hhoo_drpmd_omp
use iso_c_binding, only: c_int, c_double, c_char, c_null_char
use hhoo_1014_analytic_model, only: model_initialize, model_evaluate
use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
!$ use omp_lib, only: omp_get_num_threads
implicit none
private
logical, save :: ready=.false.
public :: drpmd_pes_abi_version, drpmd_pes_omp_enabled, drpmd_pes_init, drpmd_pes_eval
contains

integer(c_int) function drpmd_pes_abi_version() bind(C)
drpmd_pes_abi_version=1
end function

integer(c_int) function drpmd_pes_omp_enabled() bind(C)
drpmd_pes_omp_enabled=0
!$ drpmd_pes_omp_enabled=1
end function

integer(c_int) function drpmd_pes_init(filename) bind(C)
character(kind=c_char), intent(in) :: filename(*)
character(len=:), allocatable :: path
integer :: n,i,code
n=0
do i=1,4096
    if (filename(i)==c_null_char) exit
    n=n+1
end do
if (n==0 .or. n==4096) then
    drpmd_pes_init=91
    return
end if
allocate(character(len=n) :: path)
do i=1,n
    path(i:i)=filename(i)
end do
call model_initialize(path,code)
if (code==0) ready=.true.
drpmd_pes_init=int(code,c_int)
end function

integer(c_int) function drpmd_pes_eval(natoms,nbeads,q,v,g,nthreads,status,used_threads) bind(C)
integer(c_int), value, intent(in) :: natoms,nbeads,nthreads
real(c_double), intent(in) :: q(3,natoms,nbeads)
real(c_double), intent(out) :: v(nbeads),g(3,natoms,nbeads)
integer(c_int), intent(out) :: status(nbeads),used_threads
integer :: bead,code
used_threads=1
drpmd_pes_eval=1
if (natoms<1 .or. nbeads<1 .or. nthreads<1) return
v=ieee_value(0d0,ieee_quiet_nan)
g=ieee_value(0d0,ieee_quiet_nan)
status=90
if (.not.ready) return
status=1
if (natoms/=4) return
! Fixed bead slices preserve the scalar kernel's floating-point operation order.
! All evaluation temporaries belong to RECURSIVE routines in the original PES.
!$omp parallel default(none) num_threads(nthreads) &
!$omp shared(q,v,g,status,natoms,nbeads,used_threads) private(bead,code)
!$omp single
!$ used_threads=omp_get_num_threads()
!$omp end single
!$omp do schedule(static)
do bead=1,nbeads
    call model_evaluate(q(:,:,bead),v(bead),g(:,:,bead),code)
    status(bead)=int(code,c_int)
end do
!$omp end do
!$omp end parallel
drpmd_pes_eval=0
if (any(status/=0)) drpmd_pes_eval=2
end function
end module
