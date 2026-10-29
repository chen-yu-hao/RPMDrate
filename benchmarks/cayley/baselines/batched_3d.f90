!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!   RPMDrate - Bimolecular reaction rates via ring polymer molecular dynamics
!
!   Copyright (c) 2012 by Joshua W. Allen (jwallen@mit.edu)
!                         William H. Green (whgreen@mit.edu)
!                         Yury V. Suleimanov (ysuleyma@mit.edu, ysuleyma@princeton.edu)
!
!   Permission is hereby granted, free of charge, to any person obtaining a
!   copy of this software and associated documentation files (the "Software"),
!   to deal in the Software without restriction, including without limitation
!   the rights to use, copy, modify, merge, publish, distribute, sublicense,
!   and/or sell copies of the Software, and to permit persons to whom the
!   Software is furnished to do so, subject to the following conditions:
!
!   The above copyright notice and this permission notice shall be included in
!   all copies or substantial portions of the Software.
!
!   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
!   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
!   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
!   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
!   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
!   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
!   DEALINGS IN THE SOFTWARE.
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! Frozen Cayley benchmark baseline: batched_3d
module system

    implicit none

    integer, parameter :: MAX_ATOMS = 100
    double precision :: dt
    double precision :: beta
    double precision :: mass(MAX_ATOMS)
    integer :: mode
    double precision :: pi = dacos(-1.0d0)
    double precision, allocatable :: CC(:,:)
    double precision, allocatable :: ICC(:,:)
    integer :: ALLCC = 0
    ! Cached factors for the cyclic-tridiagonal Cayley free-ring propagator.
    ! The factorization depends only on beta, dt, and Nbeads, not on atom mass.
    integer :: CAYLEY_NBEADS = 0
    logical :: CAYLEY_READY = .false.
    double precision :: CAYLEY_DT = -1.0d0
    double precision :: CAYLEY_BETA = -1.0d0
    double precision :: CAYLEY_OFF = 0.0d0
    double precision :: CAYLEY_S11 = 1.0d0
    double precision :: CAYLEY_S12 = 0.0d0
    double precision :: CAYLEY_S21 = 0.0d0
    double precision :: CAYLEY_S22 = 1.0d0
    double precision :: CAYLEY_SINV11 = 1.0d0
    double precision :: CAYLEY_SINV12 = 0.0d0
    double precision :: CAYLEY_SINV21 = 0.0d0
    double precision :: CAYLEY_SINV22 = 1.0d0
    double precision, allocatable :: CAYLEY_INV_DIAG(:)
    double precision, allocatable :: CAYLEY_L(:)
    double precision, allocatable :: CAYLEY_Z1(:)
    double precision, allocatable :: CAYLEY_Z2(:)
    double precision, allocatable :: CAYLEY_WORK(:,:,:)
    ! The type of thermostat (1 = Andersen, 2 = GLE)
    integer :: thermostat

    ! Parameters for the Andersen thermostat
    double precision :: andersen_sampling_time

    ! Parameters for the GLE thermostat
    integer, parameter :: MAX_GLE_NS = 20
    integer :: gle_Ns
    double precision :: gle_A(MAX_GLE_NS+1,MAX_GLE_NS+1)
    double precision :: gle_C(MAX_GLE_NS+1,MAX_GLE_NS+1)
    double precision, allocatable, save :: gle_S(:,:), gle_T(:,:)
    double precision, allocatable, save :: gle_p(:,:,:,:), gle_np(:,:,:,:)

contains
    subroutine cayley_prepare(Nbeads)
        implicit none
        integer, intent(in) :: Nbeads
        integer :: k
        double precision :: alpha, off, det
        double precision, allocatable :: rhs(:)

        if (Nbeads .le. 0) return
        if (CAYLEY_READY .and. CAYLEY_NBEADS .eq. Nbeads .and. &
            CAYLEY_DT .eq. dt .and. CAYLEY_BETA .eq. beta) return

        if (allocated(CAYLEY_INV_DIAG)) deallocate(CAYLEY_INV_DIAG)
        if (allocated(CAYLEY_L)) deallocate(CAYLEY_L)
        if (allocated(CAYLEY_Z1)) deallocate(CAYLEY_Z1)
        if (allocated(CAYLEY_Z2)) deallocate(CAYLEY_Z2)

        CAYLEY_READY = .false.
        CAYLEY_NBEADS = Nbeads
        CAYLEY_DT = dt
        CAYLEY_BETA = beta

        if (Nbeads .lt. 3) then
            CAYLEY_READY = .true.
            return
        end if

        alpha = 0.25d0 * (dt * dble(Nbeads) / beta)**2
        off = -alpha
        CAYLEY_OFF = off
        allocate(CAYLEY_INV_DIAG(Nbeads), CAYLEY_L(Nbeads))
        allocate(CAYLEY_Z1(Nbeads), CAYLEY_Z2(Nbeads))

        ! LDL-like Thomas factors for the open tridiagonal matrix T.
        CAYLEY_INV_DIAG(1) = 1.0d0 / (1.0d0 + 2.0d0 * alpha)
        CAYLEY_L(1) = 0.0d0
        do k = 2, Nbeads
            CAYLEY_L(k) = off * CAYLEY_INV_DIAG(k-1)
            CAYLEY_INV_DIAG(k) = 1.0d0 / (1.0d0 + 2.0d0 * alpha - CAYLEY_L(k) * off)
        end do

        allocate(rhs(Nbeads))
        rhs = 0.0d0
        rhs(1) = off
        call cayley_solve_tridiagonal(rhs, CAYLEY_Z1, Nbeads)
        rhs = 0.0d0
        rhs(Nbeads) = off
        call cayley_solve_tridiagonal(rhs, CAYLEY_Z2, Nbeads)
        deallocate(rhs)

        ! M = T + U V^T, with U=(off*e_1, off*e_P) and
        ! V=(e_P, e_1).  The 2x2 Woodbury system is I + V^T T^{-1} U.
        CAYLEY_S11 = 1.0d0 + CAYLEY_Z1(Nbeads)
        CAYLEY_S12 = CAYLEY_Z2(Nbeads)
        CAYLEY_S21 = CAYLEY_Z1(1)
        CAYLEY_S22 = 1.0d0 + CAYLEY_Z2(1)
        det = CAYLEY_S11 * CAYLEY_S22 - CAYLEY_S12 * CAYLEY_S21
        CAYLEY_SINV11 = CAYLEY_S22 / det
        CAYLEY_SINV12 = -CAYLEY_S12 / det
        CAYLEY_SINV21 = -CAYLEY_S21 / det
        CAYLEY_SINV22 = CAYLEY_S11 / det
        CAYLEY_READY = .true.
    end subroutine cayley_prepare
    subroutine cayley_solve_tridiagonal(rhs, x, Nbeads)
        implicit none
        integer, intent(in) :: Nbeads
        double precision, intent(in) :: rhs(Nbeads)
        double precision, intent(out) :: x(Nbeads)
        double precision :: y(Nbeads)
        integer :: k

        y(1) = rhs(1)
        do k = 2, Nbeads
            y(k) = rhs(k) - CAYLEY_L(k) * y(k-1)
        end do
        x(Nbeads) = y(Nbeads) * CAYLEY_INV_DIAG(Nbeads)
        do k = Nbeads-1, 1, -1
            x(k) = (y(k) - CAYLEY_OFF * x(k+1)) * CAYLEY_INV_DIAG(k)
        end do
    end subroutine cayley_solve_tridiagonal
    subroutine cayley_solve_cyclic(x, Natoms, Nbeads)
        implicit none
        integer, intent(in) :: Natoms, Nbeads
        double precision, intent(inout) :: x(3,Natoms,Nbeads+2)
        integer :: k

        do k = 2, Nbeads
            x(:,:,k) = x(:,:,k) - CAYLEY_L(k) * x(:,:,k-1)
        end do
        x(:,:,Nbeads) = x(:,:,Nbeads) * CAYLEY_INV_DIAG(Nbeads)
        do k = Nbeads-1, 1, -1
            x(:,:,k) = (x(:,:,k) - CAYLEY_OFF * x(:,:,k+1)) * CAYLEY_INV_DIAG(k)
        end do
        x(:,:,Nbeads+1) = CAYLEY_SINV11 * x(:,:,Nbeads) + CAYLEY_SINV12 * x(:,:,1)
        x(:,:,Nbeads+2) = CAYLEY_SINV21 * x(:,:,Nbeads) + CAYLEY_SINV22 * x(:,:,1)
        do k = 1, Nbeads
            x(:,:,k) = x(:,:,k) - CAYLEY_Z1(k) * x(:,:,Nbeads+1) - CAYLEY_Z2(k) * x(:,:,Nbeads+2)
        end do
    end subroutine cayley_solve_cyclic
    subroutine free_ring_polymer_cayley_step(p, q, Natoms, Nbeads)
        implicit none
        integer, intent(in) :: Natoms, Nbeads
        double precision, intent(inout) :: p(3,Natoms,Nbeads), q(3,Natoms,Nbeads)
        double precision :: half_dt_mass(MAX_ATOMS), spring_mass(MAX_ATOMS)
        double precision :: alpha, wn, b1, b2, mid1, mid2, scale, impulse
        integer :: i, j, k, km, kp

        if (Nbeads .le. 0) return
        if (Nbeads .eq. 1) then
            do i = 1, 3
                do j = 1, Natoms
                    q(i,j,1) = q(i,j,1) + dt * p(i,j,1) / mass(j)
                end do
            end do
            return
        end if

        call cayley_prepare(Nbeads)
        wn = dble(Nbeads) / beta
        alpha = 0.25d0 * (dt * wn)**2
        do j = 1, Natoms
            half_dt_mass(j) = 0.5d0 * dt / mass(j)
            spring_mass(j) = dt * mass(j) * wn * wn
        end do

        if (Nbeads .eq. 2) then
            scale = 1.0d0 / (1.0d0 + 4.0d0 * alpha)
            do j = 1, Natoms
                do i = 1, 3
                    b1 = q(i,j,1) + half_dt_mass(j) * p(i,j,1)
                    b2 = q(i,j,2) + half_dt_mass(j) * p(i,j,2)
                    mid1 = ((1.0d0 + 2.0d0 * alpha) * b1 + 2.0d0 * alpha * b2) * scale
                    mid2 = (2.0d0 * alpha * b1 + (1.0d0 + 2.0d0 * alpha) * b2) * scale
                    impulse = 2.0d0 * spring_mass(j) * (mid1 - mid2)
                    p(i,j,1) = p(i,j,1) - impulse
                    p(i,j,2) = p(i,j,2) + impulse
                    q(i,j,1) = 2.0d0 * mid1 - q(i,j,1)
                    q(i,j,2) = 2.0d0 * mid2 - q(i,j,2)
                end do
            end do
            return
        end if

        if (allocated(CAYLEY_WORK)) then
            if (size(CAYLEY_WORK,2) .ne. Natoms .or. size(CAYLEY_WORK,3) .ne. Nbeads+2) deallocate(CAYLEY_WORK)
        end if
        if (.not.allocated(CAYLEY_WORK)) allocate(CAYLEY_WORK(3,Natoms,Nbeads+2))

        ! The same Cayley map in midpoint form: M q_mid = q + dt*p/(2*m),
        ! q_new = 2*q_mid - q, p_new = p - dt*m*omega_P**2*L*q_mid.
        do k = 1, Nbeads
            do j = 1, Natoms
                CAYLEY_WORK(:,j,k) = q(:,j,k) + half_dt_mass(j) * p(:,j,k)
            end do
        end do
        call cayley_solve_cyclic(CAYLEY_WORK, Natoms, Nbeads)
        do k = 1, Nbeads
            km = k - 1
            if (km .lt. 1) km = Nbeads
            kp = k + 1
            if (kp .gt. Nbeads) kp = 1
            do j = 1, Natoms
                p(:,j,k) = p(:,j,k) - spring_mass(j) * &
                    (2.0d0 * CAYLEY_WORK(:,j,k) - CAYLEY_WORK(:,j,km) - CAYLEY_WORK(:,j,kp))
                q(:,j,k) = 2.0d0 * CAYLEY_WORK(:,j,k) - q(:,j,k)
            end do
        end do
    end subroutine free_ring_polymer_cayley_step
end module system
