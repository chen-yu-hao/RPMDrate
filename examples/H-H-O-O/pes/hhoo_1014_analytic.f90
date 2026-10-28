! HHOO potential with analytic Cartesian energy gradients.
! Atom order: H, H, O, O.  Coordinates: bohr.  Energy: hartree.
! g and dVdq are +dV/dx in hartree/bohr; the physical force is -g.
!
! External interfaces (ordinary Fortran calling convention):
!   hhoo_1014_init(weights_path, result)       ! explicit serial initialization
!   initialize_potential()                   ! serial default: ./txt_1014
!   hhoo_1014_grad(x(3,4), v, g(3,4))         ! error stop on failure
!   hhoo_1014(x(3,4), v)                      ! compatible energy-only interface
!   get_potential(q, V, dVdq, Natoms, Nbeads, result)
!     q/dVdq: (3,Natoms,Nbeads); V: (Nbeads); Natoms must be 4.
!
! Call hhoo_1014_init or initialize_potential ONCE, SERIALLY before concurrent
! evaluations.  Existing serial callers may omit it: their first evaluation
! loads ./txt_1014.  Concurrent first-time initialization is not supported.
! After successful initialization all model data are read-only.  Numerical
! routines are RECURSIVE with local work arrays; no OpenMP parallelism is added.
! The loaded model cannot be replaced.  Repeating explicit initialization with
! the same literal path succeeds; a different path returns status 13.
! An empty explicit path selects txt_1014.  A failed load may be retried.
!
! Status codes: 0 success; 1 Natoms /= 4; 2 Nbeads < 1; 3 invalid geometry;
!   10 weight-file open failure; 11 read/close failure; 12 invalid weights/scales;
!   13 attempted replacement of initialized weights; 20 non-finite result.
! Batch failures return nonzero result and NaN in every output element.
! Interfaces without a result argument print the status and terminate on error.
!
! Mathematical kernel: the independently validated analytic_pes.f90 control.
! Its forward operation order, five-network averaging, descriptors, activation
! derivatives, and Cartesian chain rule are retained without algebraic changes.
! Compile with normal FP64 semantics; do not enable unsafe fast-math options.

module hhoo_1014_analytic_model
use iso_c_binding, only: c_double
use, intrinsic :: ieee_arithmetic, only: ieee_is_finite, ieee_value, ieee_quiet_nan
implicit none
private

integer, parameter :: l0=7,l1=50,l2=50,ln=5
integer, parameter, public :: HHOO_SUCCESS=0, HHOO_BAD_NATOMS=1, HHOO_BAD_NBEADS=2
integer, parameter, public :: HHOO_BAD_GEOMETRY=3, HHOO_OPEN_ERROR=10, HHOO_READ_ERROR=11
integer, parameter, public :: HHOO_BAD_WEIGHTS=12, HHOO_MODEL_FROZEN=13, HHOO_NONFINITE=20
logical, save :: initialized=.false.
character(len=:), allocatable, save :: loaded_path
real(c_double), save :: w1(l0,l1,ln),b1(l1,ln),w2(l1,l2,ln),b2(l2,ln),w3(l2,ln),b3(ln)
real(c_double), save :: scile(2,l0,ln),scole(2,ln)

public :: model_initialize, model_ensure_initialized, model_evaluate

contains

recursive subroutine model_initialize(weights_path,result)
character(len=*), intent(in) :: weights_path
integer, intent(out) :: result
character(len=:), allocatable :: path
integer :: n,i,unit,ios,close_ios,header(5)
integer, parameter :: expected_header(5)=[2,7,50,50,3001]
real(c_double) :: tw1(l0,l1,ln),tb1(l1,ln),tw2(l1,l2,ln),tb2(l2,ln),tw3(l2,ln),tb3(ln)
real(c_double) :: tscile(2,l0,ln),tscole(2,ln),width(l0,ln),output_width(ln)
real(c_double) :: nan_value

result=HHOO_SUCCESS
if (len_trim(weights_path)==0) then
    path='txt_1014'
else
    path=trim(weights_path)
end if
if (initialized) then
    if (path/=loaded_path) result=HHOO_MODEL_FROZEN
    return
end if

! Read into local temporary arrays.  No partially loaded model is published.
! List-directed null values or '/' can leave entries unchanged with iostat=0.
! NaN sentinels ensure the finite-value validation rejects incomplete data.
nan_value=ieee_value(0d0,ieee_quiet_nan)
tw1=nan_value
tb1=nan_value
tw2=nan_value
tb2=nan_value
tw3=nan_value
tb3=nan_value
tscile=nan_value
tscole=nan_value
open(newunit=unit,file=path,status='old',action='read',iostat=ios)
if (ios/=0) then
    result=HHOO_OPEN_ERROR
    return
end if
do n=1,ln
    header=0
    do i=1,5
        read(unit,*,iostat=ios)header(i)
        if (ios/=0) go to 900
    end do
    if (any(header/=expected_header)) then
        close(unit,iostat=close_ios)
        result=HHOO_BAD_WEIGHTS
        return
    end if
    read(unit,*,iostat=ios)tscile(:,:,n),tscole(:,n)
    if (ios/=0) go to 900
    do i=1,l1+l2+1
        read(unit,*,iostat=ios)
        if (ios/=0) go to 900
    end do
    do i=1,l1
        read(unit,*,iostat=ios)tw1(:,i,n),tb1(i,n)
        if (ios/=0) go to 900
    end do
    do i=1,l2
        read(unit,*,iostat=ios)tw2(:,i,n),tb2(i,n)
        if (ios/=0) go to 900
    end do
    read(unit,*,iostat=ios)tw3(:,n),tb3(n)
    if (ios/=0) go to 900
end do
close(unit,iostat=ios)
if (ios/=0) then
    result=HHOO_READ_ERROR
    return
end if

if (.not.all(ieee_is_finite(tw1)) .or. .not.all(ieee_is_finite(tb1)) .or. &
    .not.all(ieee_is_finite(tw2)) .or. .not.all(ieee_is_finite(tb2)) .or. &
    .not.all(ieee_is_finite(tw3)) .or. .not.all(ieee_is_finite(tb3)) .or. &
    .not.all(ieee_is_finite(tscile)) .or. .not.all(ieee_is_finite(tscole))) then
    result=HHOO_BAD_WEIGHTS
    return
end if
width=tscile(2,:,:)-tscile(1,:,:)
output_width=tscole(2,:)-tscole(1,:)
if (.not.all(ieee_is_finite(width)) .or. any(width==0d0) .or. &
    .not.all(ieee_is_finite(output_width))) then
    result=HHOO_BAD_WEIGHTS
    return
end if

! This commit must be performed by the caller's serial initialization phase.
w1=tw1
b1=tb1
w2=tw2
b2=tb2
w3=tw3
b3=tb3
scile=tscile
scole=tscole
loaded_path=path
initialized=.true.
return

900 continue
close(unit,iostat=close_ios)
result=HHOO_READ_ERROR
end subroutine model_initialize

recursive subroutine model_ensure_initialized(result)
integer, intent(out) :: result
result=HHOO_SUCCESS
if (.not.initialized) call model_initialize('txt_1014',result)
end subroutine model_ensure_initialized

recursive subroutine model_evaluate(x,v,g,result)
real(c_double), intent(in) :: x(3,4)
real(c_double), intent(out) :: v,g(3,4)
integer, intent(out) :: result
real(c_double) :: delta_check(3),r2_check,nan_value
integer :: i,j

nan_value=ieee_value(0d0,ieee_quiet_nan)
v=nan_value
g=nan_value
call model_ensure_initialized(result)
if (result/=HHOO_SUCCESS) return
if (.not.all(ieee_is_finite(x))) then
    result=HHOO_BAD_GEOMETRY
    return
end if
do i=1,4
    do j=i+1,4
        delta_check=x(:,i)-x(:,j)
        if (.not.all(ieee_is_finite(delta_check))) then
            result=HHOO_BAD_GEOMETRY
            return
        end if
        r2_check=sum(delta_check**2d0)
        if (.not.ieee_is_finite(r2_check) .or. r2_check<=0d0) then
            result=HHOO_BAD_GEOMETRY
            return
        end if
    end do
end do

call energy_gradient(x,v,g)
if (.not.ieee_is_finite(v) .or. .not.all(ieee_is_finite(g))) then
    result=HHOO_NONFINITE
    v=nan_value
    g=nan_value
    return
end if
result=HHOO_SUCCESS
end subroutine model_evaluate

recursive subroutine network_energy_gradient(p,v,dvdp)
real(c_double), intent(in) :: p(7)
real(c_double), intent(out) :: v,dvdp(7)
real(c_double) :: a0(l0),a1(l1),a2(l2),a3,den1(l1),den2(l2)
real(c_double) :: adj0(l0),adj1(l1),adj2(l2),output_factor
integer :: n,i,j
v=0d0
dvdp=0d0
do n=1,ln
    a0=(p-scile(1,:,n))/(scile(2,:,n)-scile(1,:,n))*2d0-1d0
    a1=b1(:,n)
    do i=1,l1
        do j=1,l0
            a1(i)=a1(i)+w1(j,i,n)*a0(j)
        end do
    end do
    den1=dsqrt(a1**2d0+1d0)
    a1=a1/den1
    a2=b2(:,n)
    do i=1,l2
        do j=1,l1
            a2(i)=a2(i)+w2(j,i,n)*a1(j)
        end do
    end do
    den2=dsqrt(a2**2d0+1d0)
    a2=a2/den2
    a3=b3(n)+dot_product(w3(:,n),a2)
    v=v+(a3*(scole(2,n)-scole(1,n))+scole(1,n)+scole(2,n))/2d0/dble(ln)

    output_factor=(scole(2,n)-scole(1,n))/(2d0*dble(ln))
    adj2=w3(:,n)*output_factor/(den2**3)
    adj1=0d0
    do i=1,l2
        do j=1,l1
            adj1(j)=adj1(j)+w2(j,i,n)*adj2(i)
        end do
    end do
    adj1=adj1/(den1**3)
    adj0=0d0
    do i=1,l1
        do j=1,l0
            adj0(j)=adj0(j)+w1(j,i,n)*adj1(i)
        end do
    end do
    dvdp=dvdp+adj0*2d0/(scile(2,:,n)-scile(1,:,n))
end do
end subroutine network_energy_gradient

recursive subroutine energy_gradient(x,v,g)
real(c_double), intent(in) :: x(3,4)
real(c_double), intent(out) :: v,g(3,4)
real(c_double) :: rho(6),delta(3,6),p(7),dvdp(7),dvdrho(6),pair_gradient(3)
integer :: i,j,k
k=0
do i=1,4
    do j=i+1,4
        k=k+1
        delta(:,k)=x(:,i)-x(:,j)
        rho(k)=1d0/dsqrt(sum(delta(:,k)**2d0))
    end do
end do
p(1)=rho(2)+rho(3)+rho(4)+rho(5)
p(2)=rho(2)**2d0+rho(3)**2d0+rho(4)**2d0+rho(5)**2d0
p(3)=rho(2)*rho(3)+rho(4)*rho(5)
p(4)=rho(2)*rho(4)+rho(3)*rho(5)
p(5)=rho(2)**3d0+rho(3)**3d0+rho(4)**3d0+rho(5)**3d0
p(6)=rho(1)
p(7)=rho(6)
p(2:4)=p(2:4)**(1d0/2d0)
p(5)=p(5)**(1d0/3d0)
call network_energy_gradient(p,v,dvdp)

dvdrho(1)=dvdp(6)
dvdrho(6)=dvdp(7)
do k=2,5
    dvdrho(k)=dvdp(1)+dvdp(2)*rho(k)/p(2)+dvdp(5)*rho(k)**2/p(5)**2
end do
dvdrho(2)=dvdrho(2)+dvdp(3)*rho(3)/(2d0*p(3))+dvdp(4)*rho(4)/(2d0*p(4))
dvdrho(3)=dvdrho(3)+dvdp(3)*rho(2)/(2d0*p(3))+dvdp(4)*rho(5)/(2d0*p(4))
dvdrho(4)=dvdrho(4)+dvdp(3)*rho(5)/(2d0*p(3))+dvdp(4)*rho(2)/(2d0*p(4))
dvdrho(5)=dvdrho(5)+dvdp(3)*rho(4)/(2d0*p(3))+dvdp(4)*rho(3)/(2d0*p(4))
g=0d0
k=0
do i=1,4
    do j=i+1,4
        k=k+1
        pair_gradient=-dvdrho(k)*rho(k)**3*delta(:,k)
        g(:,i)=g(:,i)+pair_gradient
        g(:,j)=g(:,j)-pair_gradient
    end do
end do
end subroutine energy_gradient

end module hhoo_1014_analytic_model


recursive subroutine hhoo_1014_init(weights_path,result)
use hhoo_1014_analytic_model, only: model_initialize
implicit none
character(len=*), intent(in) :: weights_path
integer, intent(out) :: result
call model_initialize(weights_path,result)
end subroutine hhoo_1014_init


recursive subroutine initialize_potential()
use iso_fortran_env, only: error_unit
use hhoo_1014_analytic_model, only: model_ensure_initialized, HHOO_SUCCESS
implicit none
integer :: result
call model_ensure_initialized(result)
if (result/=HHOO_SUCCESS) then
    write(error_unit,'(a,i0)') 'initialize_potential: HHOO status ',result
    error stop 1
end if
end subroutine initialize_potential


recursive subroutine hhoo_1014_grad(x,v,g)
use iso_c_binding, only: c_double
use iso_fortran_env, only: error_unit
use hhoo_1014_analytic_model, only: model_evaluate, HHOO_SUCCESS
implicit none
real(c_double), intent(in) :: x(3,4)
real(c_double), intent(out) :: v,g(3,4)
integer :: result
call model_evaluate(x,v,g,result)
if (result/=HHOO_SUCCESS) then
    write(error_unit,'(a,i0)') 'hhoo_1014_grad: HHOO status ',result
    error stop 1
end if
end subroutine hhoo_1014_grad


recursive subroutine hhoo_1014(x,v)
use iso_c_binding, only: c_double
implicit none
real(c_double), intent(in) :: x(3,4)
real(c_double), intent(out) :: v
real(c_double) :: g(3,4)
call hhoo_1014_grad(x,v,g)
end subroutine hhoo_1014


recursive subroutine get_potential(q,V,dVdq,Natoms,Nbeads,result)
use iso_c_binding, only: c_double
use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
use hhoo_1014_analytic_model, only: model_ensure_initialized, model_evaluate, &
    HHOO_SUCCESS, HHOO_BAD_NATOMS, HHOO_BAD_NBEADS
implicit none
integer, intent(in) :: Natoms,Nbeads
real(c_double), intent(in) :: q(3,Natoms,Nbeads)
real(c_double), intent(out) :: V(Nbeads),dVdq(3,Natoms,Nbeads)
integer, intent(out) :: result
integer :: bead
real(c_double) :: nan_value

nan_value=ieee_value(0d0,ieee_quiet_nan)
V=nan_value
dVdq=nan_value
if (Natoms/=4) then
    result=HHOO_BAD_NATOMS
    return
end if
if (Nbeads<1) then
    result=HHOO_BAD_NBEADS
    return
end if
call model_ensure_initialized(result)
if (result/=HHOO_SUCCESS) return
do bead=1,Nbeads
    call model_evaluate(q(:,:,bead),V(bead),dVdq(:,:,bead),result)
    if (result/=HHOO_SUCCESS) then
        ! Discard partial output from earlier beads on any batch failure.
        V=nan_value
        dVdq=nan_value
        return
    end if
end do
result=HHOO_SUCCESS
end subroutine get_potential
