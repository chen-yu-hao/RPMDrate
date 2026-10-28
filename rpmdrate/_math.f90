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

! Seed the pseudo-random number generator using the system clock. The
! implementation is based on an example in the gfortran documentation for
! the intrinsic random_seed() subroutine. Note that multiple calls to this
! subroutine do not reseed the pseudo-random number generator.
subroutine random_init()
    implicit none
    integer :: i, n, clock
    integer, dimension(:), allocatable :: seed

    call random_seed(size=n)
    allocate(seed(n))

    call system_clock(count=clock)

    seed = clock + 37 * (/ (i - 1, i = 1, n) /)
    call random_seed(put = seed)

    deallocate(seed)

end subroutine random_init

! Seed the pseudo-random number generator using a particular seed value.
! Useful for replicating runs.
subroutine random_init_seed(value)
    implicit none
    integer, intent(in) :: value
    integer, dimension(:), allocatable :: seed
    integer :: n, i

    call random_seed(size=n)
    allocate(seed(n))

    do i = 1, n
        seed(i) = abs(value) + (i - 1)
    end do
    call random_seed(put = seed)

    deallocate(seed)

end subroutine random_init_seed

! Compute a pseudo-random number uniformly distributed in [0,1].
! Returns:
!   rn - The pseudo-random number
subroutine random(rn)
    implicit none
    double precision, intent(out) :: rn
    call random_number(rn)
end subroutine random

! Compute a pseudo-random number weighted by a standard normal distribution.
! The Marsaglia polar method is used to convert from a uniform distribution to
! a normal distribution.
! Returns:
!   rn - The pseudo-random number
subroutine randomn(rn)

    implicit none
    double precision, intent(out) :: rn
    integer :: iset
    double precision :: gset, u, v, S, fac

    save iset,gset

    data iset/0/

    ! The Marsaglia polar method
    if (iset .eq. 0) then
        S = 1.d0
        do while (S .ge. 1.d0 .or. S .eq. 0.d0)
            call random(u)
            call random(v)
            u = 2.d0 * u - 1.d0
            v = 2.d0 * v - 1.d0
            S = u * u + v * v
        end do
        fac = sqrt(-2 * log(S) / S)
        gset = u * fac
        rn = v * fac
        iset = 1
    else
        rn = gset
        iset = 0
    end if

end subroutine randomn

! Compute the real Fourier transform of the given array of data.
! This fallback keeps the legacy half-complex API without requiring FFTW;
! production free-ring propagation now uses the Cayley solver in _main.f90.
! Parameters:
!   x - The array of data to transform
!   N - The length of the array of data
! Returns:
!   x - The transformed array of data, in half-complex form
subroutine rfft(x,N)

    implicit none
    integer, intent(in) :: N
    double precision, intent(inout) :: x(N)

    integer :: k, j, nhalf
    double precision :: factor, angle
    complex(kind=8) :: z(N), phase

    factor = 1.d0 / dsqrt(dble(N))
    z = (0.d0, 0.d0)
    do k = 0, N - 1
        do j = 0, N - 1
            angle = -2.d0 * dacos(-1.d0) * dble(k*j) / dble(N)
            phase = cmplx(dcos(angle), dsin(angle), kind=8)
            z(k+1) = z(k+1) + cmplx(x(j+1), 0.d0, kind=8) * phase
        end do
    end do

    x = 0.d0
    x(1) = factor * dble(z(1))
    nhalf = (N - 1) / 2
    do k = 1, nhalf
        x(k+1) = factor * dble(z(k+1))
        x(N-k+1) = factor * dimag(z(k+1))
    end do
    if (mod(N,2) .eq. 0) x(N/2+1) = factor * dble(z(N/2+1))

end subroutine rfft

! Compute the inverse real Fourier transform of half-complex data.
! Parameters:
!   x - The array of data to transform, in half-complex form
!   N - The length of the array of data
! Returns:
!   x - The transformed array of data
subroutine irfft(x,N)

    implicit none
    integer, intent(in) :: N
    double precision, intent(inout) :: x(N)

    integer :: k, j, nhalf
    double precision :: factor, angle
    complex(kind=8) :: z(N), sumz, phase

    factor = 1.d0 / dsqrt(dble(N))
    z = (0.d0, 0.d0)
    z(1) = cmplx(x(1), 0.d0, kind=8)
    nhalf = (N - 1) / 2
    do k = 1, nhalf
        z(k+1) = cmplx(x(k+1), x(N-k+1), kind=8)
        z(N-k+1) = dconjg(z(k+1))
    end do
    if (mod(N,2) .eq. 0) z(N/2+1) = cmplx(x(N/2+1), 0.d0, kind=8)

    do j = 0, N - 1
        sumz = (0.d0, 0.d0)
        do k = 0, N - 1
            angle = 2.d0 * dacos(-1.d0) * dble(k*j) / dble(N)
            phase = cmplx(dcos(angle), dsin(angle), kind=8)
            sumz = sumz + z(k+1) * phase
        end do
        x(j+1) = factor * dble(sumz)
    end do

end subroutine irfft

! Return the expoential of a (square) matrix using the scale and square
! algorithm.
! Parameters:
!   M - The square matrix to exponentiate
!   n - The size of the matrix
!   j - The number of terms in the exponential to evaluate
!   k - The exponent of the power of two used to scale the matrix
! Returns:
!   EM - The expoential of the given matrix
subroutine matrix_exp(M, n, j, k, EM)
    integer, intent(in)  :: n, j, k
    real*8, intent(in)   :: M(n,n)
    real*8, intent(out)   :: EM(n,n)

    real *8 :: tc(j+1), SM(n,n)
    integer p, i
    tc(1)=1
    do i=1,j
       tc(i+1)=tc(i)/dble(i)
    enddo

    !scale
    SM=M*(1./2.**k)
    EM=0.
    do i=1,n
       EM(i,i)=tc(j+1)
    enddo

    !taylor exp of scaled matrix
    do p=j,1,-1
       EM=matmul(SM,EM);
       do i=1,n
          EM(i,i)=EM(i,i)+tc(p)
       enddo
    enddo

    !square
    do p=1,k
       EM=matmul(EM,EM)
    enddo

end subroutine matrix_exp

! Compute the brute-force stabilized Cholesky decomposition of a square matrix.
! Parameters:
!   SST - The matrix to determine the Cholesky decomposition of
!   n - The size of the matrix
! Returns:
!   S - The computed Cholesky decomposition
subroutine cholesky(SST, S, n)

    integer, intent(in)  :: n
    real*8, intent(in)   :: SST(n,n)
    real*8, intent(out)   :: S(n,n)
    real*8 :: D(n), L(n,n)
    
    integer i,j,k
    
    D = 0.d0
    L = 0.d0
    do i = 1,n
        L(i,i)=1.
        D(i)=SST(i,i)
        do j=1,i-1
            L(i,j)=SST(i,j);
            do k=1,j-1
                L(i,j)=L(i,j)-L(i,k)*L(j,k)*D(k)
            end do
            if (D(j).ne.0.) L(i,j)=L(i,j)/D(j)
        end do
        do k=1,i-1
            D(i)=D(i)-L(i,k)*L(i,k)*D(k)
        end do
    end do
    S=0.
    do i=1,n
        do j=1,i
            if (D(j)>0.) S(i,j)=S(i,j)+L(i,j)*sqrt(D(j))
        end do
    end do

end subroutine cholesky
