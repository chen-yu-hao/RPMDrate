! The supplied five-pass stepB, kept as a benchmark-only comparison.
module step_b_reference
    use system, only: dt, beta, mass, CAYLEY_INV_DIAG, CAYLEY_L, CAYLEY_OFF, &
        CAYLEY_Z1, CAYLEY_Z2, CAYLEY_SINV11, CAYLEY_SINV12, CAYLEY_SINV21, CAYLEY_SINV22
    implicit none
    double precision, allocatable :: v(:,:), r(:,:), w(:,:), m(:), imass(:), c1(:), c2(:)
    double precision :: alpha, a1, a2, pair_scale
contains
    subroutine prepare_step_b(Natoms, nb)
        integer, intent(in) :: Natoms, nb
        integer :: j, nd
        nd = 3*Natoms
        if (allocated(v)) deallocate(v, r, w, m, imass, c1, c2)
        allocate(v(nd,nb), r(nd,nb), w(nd,nb), m(nd), imass(nd), c1(nd), c2(nd))
        do j = 1, Natoms
            m(3*j-2:3*j) = mass(j)
        end do
        imass = 1.d0 / m
        alpha = .25d0 * (dt * nb / beta)**2
        a1 = 1.d0 - 2.d0 * alpha
        a2 = .5d0 * dt * (dble(nb)/beta)**2
        pair_scale = 1.d0 / (1.d0 + 4.d0 * alpha)
    end subroutine prepare_step_b

    subroutine step_b(p, q, nd, nb)
        integer, intent(in) :: nd, nb
        double precision, intent(inout) :: p(nd,nb), q(nd,nb)
        integer :: k, km, kp
        if (nb == 1) then
            q(:,1) = q(:,1) + dt * p(:,1) * imass
            return
        end if
        do k = 1, nb
            km = merge(nb, k-1, k == 1)
            kp = merge(1, k+1, k == nb)
            v(:,k) = p(:,k) * imass
            r(:,k) = a1*q(:,k) + alpha*(q(:,km) + q(:,kp)) + dt*v(:,k)
        end do
        if (nb == 2) then
            w(:,1) = q(:,1) + ((1.d0+2.d0*alpha)*r(:,1) + 2.d0*alpha*r(:,2)) * pair_scale
            w(:,2) = q(:,2) + (2.d0*alpha*r(:,1) + (1.d0+2.d0*alpha)*r(:,2)) * pair_scale
        else
            do k = 2, nb
                r(:,k) = r(:,k) - CAYLEY_L(k)*r(:,k-1)
            end do
            r(:,nb) = r(:,nb)*CAYLEY_INV_DIAG(nb)
            do k = nb-1, 1, -1
                r(:,k) = (r(:,k) - CAYLEY_OFF*r(:,k+1))*CAYLEY_INV_DIAG(k)
            end do
            c1 = CAYLEY_SINV11*r(:,nb) + CAYLEY_SINV12*r(:,1)
            c2 = CAYLEY_SINV21*r(:,nb) + CAYLEY_SINV22*r(:,1)
            do k = 1, nb
                w(:,k) = q(:,k) + r(:,k) - CAYLEY_Z1(k)*c1 - CAYLEY_Z2(k)*c2
            end do
        end if
        do k = 1, nb
            km = merge(nb, k-1, k == 1)
            kp = merge(1, k+1, k == nb)
            q(:,k) = w(:,k) - q(:,k)
            p(:,k) = m*(v(:,k) - a2*(2.d0*w(:,k) - w(:,km) - w(:,kp)))
        end do
    end subroutine step_b
end module step_b_reference
