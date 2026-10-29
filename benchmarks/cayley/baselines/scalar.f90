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

! Frozen Cayley benchmark baseline: scalar
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
    double precision, allocatable :: CAYLEY_DIAG(:)
    double precision, allocatable :: CAYLEY_L(:)
    double precision, allocatable :: CAYLEY_Z1(:)
    double precision, allocatable :: CAYLEY_Z2(:)
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

        if (allocated(CAYLEY_DIAG)) deallocate(CAYLEY_DIAG)
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
        allocate(CAYLEY_DIAG(Nbeads), CAYLEY_L(Nbeads))
        allocate(CAYLEY_Z1(Nbeads), CAYLEY_Z2(Nbeads))

        ! LDL-like Thomas factors for the open tridiagonal matrix T.
        CAYLEY_DIAG(1) = 1.0d0 + 2.0d0 * alpha
        CAYLEY_L(1) = 0.0d0
        do k = 2, Nbeads
            CAYLEY_L(k) = off / CAYLEY_DIAG(k-1)
            CAYLEY_DIAG(k) = 1.0d0 + 2.0d0 * alpha - CAYLEY_L(k) * off
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
        x(Nbeads) = y(Nbeads) / CAYLEY_DIAG(Nbeads)
        do k = Nbeads-1, 1, -1
            x(k) = (y(k) - CAYLEY_OFF * x(k+1)) / CAYLEY_DIAG(k)
        end do
    end subroutine cayley_solve_tridiagonal
    subroutine cayley_solve_cyclic(rhs, x, Nbeads)
        implicit none
        integer, intent(in) :: Nbeads
        double precision, intent(in) :: rhs(Nbeads)
        double precision, intent(out) :: x(Nbeads)
        double precision :: y(Nbeads), c1, c2, y1, y2
        integer :: k

        call cayley_solve_tridiagonal(rhs, y, Nbeads)
        y1 = y(Nbeads)
        y2 = y(1)
        c1 = CAYLEY_SINV11 * y1 + CAYLEY_SINV12 * y2
        c2 = CAYLEY_SINV21 * y1 + CAYLEY_SINV22 * y2
        do k = 1, Nbeads
            x(k) = y(k) - CAYLEY_Z1(k) * c1 - CAYLEY_Z2(k) * c2
        end do
    end subroutine cayley_solve_cyclic
    subroutine free_ring_polymer_cayley_step(p, q, Natoms, Nbeads)
        implicit none
        integer, intent(in) :: Natoms, Nbeads
        double precision, intent(inout) :: p(3,Natoms,Nbeads), q(3,Natoms,Nbeads)
        double precision :: qold(Nbeads), vold(Nbeads), rhs(Nbeads), qnew(Nbeads)
        double precision :: alpha, wn, lq, mass_j
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

        do i = 1, 3
            do j = 1, Natoms
                mass_j = mass(j)
                do k = 1, Nbeads
                    qold(k) = q(i,j,k)
                    vold(k) = p(i,j,k) / mass_j
                end do

                if (Nbeads .eq. 2) then
                    rhs(1) = (1.0d0 - 2.0d0 * alpha) * qold(1) + &
                        2.0d0 * alpha * qold(2) + dt * vold(1)
                    rhs(2) = 2.0d0 * alpha * qold(1) + &
                        (1.0d0 - 2.0d0 * alpha) * qold(2) + dt * vold(2)
                    qnew(1) = ((1.0d0 + 2.0d0 * alpha) * rhs(1) + &
                        2.0d0 * alpha * rhs(2)) / (1.0d0 + 4.0d0 * alpha)
                    qnew(2) = (2.0d0 * alpha * rhs(1) + &
                        (1.0d0 + 2.0d0 * alpha) * rhs(2)) / (1.0d0 + 4.0d0 * alpha)
                else
                    do k = 1, Nbeads
                        km = k - 1
                        if (km .lt. 1) km = Nbeads
                        kp = k + 1
                        if (kp .gt. Nbeads) kp = 1
                        rhs(k) = (1.0d0 - 2.0d0 * alpha) * qold(k) + &
                            alpha * (qold(km) + qold(kp)) + dt * vold(k)
                    end do
                    call cayley_solve_cyclic(rhs, qnew, Nbeads)
                end if

                do k = 1, Nbeads
                    km = k - 1
                    if (km .lt. 1) km = Nbeads
                    kp = k + 1
                    if (kp .gt. Nbeads) kp = 1
                    lq = 2.0d0 * (qold(k) + qnew(k)) - &
                        (qold(km) + qnew(km)) - (qold(kp) + qnew(kp))
                    q(i,j,k) = qnew(k)
                    p(i,j,k) = mass_j * (vold(k) - 0.5d0 * dt * wn * wn * lq)
                end do
            end do
        end do
    end subroutine free_ring_polymer_cayley_step
end module system
