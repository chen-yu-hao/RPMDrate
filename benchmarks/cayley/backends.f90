! Benchmark-only references. The cyclic backend calls the production system module.
module benchmark_backends
    use iso_c_binding
    use system, only: beta, dt, mass, CC, ICC, cayley_prepare, free_ring_polymer_cayley_step
    implicit none
    include 'fftw3.f03'
    real(c_double), allocatable :: coefficients(:,:,:), fft_input(:,:)
    complex(c_double_complex), allocatable :: fft_output(:,:)
    type(c_ptr) :: forward_plan = c_null_ptr, backward_plan = c_null_ptr

contains

    subroutine prepare_case(n, beta_in, dt_in, masses) bind(c)
        integer(c_int), value :: n
        real(c_double), value :: beta_in, dt_in
        real(c_double), intent(in) :: masses(4)
        integer :: k, j, atom, dimensions(1), nf
        real(c_double) :: theta, factor, w2, denominator

        beta = beta_in
        dt = dt_in
        mass(1:4) = masses
        call cayley_prepare(n)
        if (allocated(CC)) deallocate(CC, ICC)
        allocate(CC(n,n), ICC(n,n))
        factor = 1.d0 / sqrt(dble(n))
        do k = 0, n-1
            do j = 0, n-1
                theta = 2.d0 * acos(-1.d0) * dble(k*j) / n
                if (k == 0) then
                    CC(k+1,j+1) = factor
                else if (mod(n,2) == 0 .and. k == n/2) then
                    CC(k+1,j+1) = factor * (-1.d0)**j
                else if (k <= (n-1)/2) then
                    CC(k+1,j+1) = factor * cos(theta)
                else
                    CC(k+1,j+1) = factor * sin(theta)
                end if
            end do
            factor = 1.d0 / sqrt(dble(n))
            ICC(:,k+1) = 2.d0 * CC(k+1,:)
            if (k == 0 .or. (mod(n,2) == 0 .and. k == n/2)) ICC(:,k+1) = CC(k+1,:)
        end do

        if (allocated(coefficients)) deallocate(coefficients)
        allocate(coefficients(4,n,4))
        do atom = 1, 4
            do k = 0, n-1
                w2 = (2.d0 * n / beta * sin(acos(-1.d0) * min(k,n-k) / n))**2
                denominator = 1.d0 + 0.25d0 * dt**2 * w2
                coefficients(1,k+1,atom) = (1.d0 - 0.25d0 * dt**2 * w2) / denominator
                coefficients(2,k+1,atom) = -mass(atom) * dt * w2 / denominator
                coefficients(3,k+1,atom) = dt / (mass(atom) * denominator)
                coefficients(4,k+1,atom) = coefficients(1,k+1,atom)
            end do
        end do

        if (c_associated(forward_plan)) call fftw_destroy_plan(forward_plan)
        if (c_associated(backward_plan)) call fftw_destroy_plan(backward_plan)
        if (allocated(fft_input)) deallocate(fft_input, fft_output)
        nf = n/2 + 1
        allocate(fft_input(n,24), fft_output(nf,24))
        dimensions = [n]
        ! One batch contains the 12 scalar momentum rings and 12 position rings.
        ! Plan creation is excluded from timing; all packing/unpacking is timed.
        forward_plan = fftw_plan_many_dft_r2c(1, dimensions, 24, fft_input, [n], 1, n, &
            fft_output, [nf], 1, nf, FFTW_MEASURE)
        backward_plan = fftw_plan_many_dft_c2r(1, dimensions, 24, fft_output, [nf], 1, nf, &
            fft_input, [n], 1, n, FFTW_MEASURE)
        if (.not.c_associated(forward_plan) .or. .not.c_associated(backward_plan)) stop 4
    end subroutine prepare_case

    subroutine dense_cached(p, q, n)
        integer, intent(in) :: n
        real(c_double), intent(inout) :: p(3,4,n), q(3,4,n)
        real(c_double) :: p_new
        integer :: i, j, k
        do j = 1, 4
            do i = 1, 3
                p(i,j,:) = matmul(CC, p(i,j,:))
                q(i,j,:) = matmul(CC, q(i,j,:))
            end do
        end do
        do j = 1, 4
            do k = 1, n
                do i = 1, 3
                    p_new = coefficients(1,k,j)*p(i,j,k) + coefficients(2,k,j)*q(i,j,k)
                    q(i,j,k) = coefficients(3,k,j)*p(i,j,k) + coefficients(4,k,j)*q(i,j,k)
                    p(i,j,k) = p_new
                end do
            end do
        end do
        do j = 1, 4
            do i = 1, 3
                p(i,j,:) = matmul(ICC, p(i,j,:))
                q(i,j,:) = matmul(ICC, q(i,j,:))
            end do
        end do
    end subroutine dense_cached

    subroutine fftw_cayley(p, q, n)
        integer, intent(in) :: n
        real(c_double), intent(inout) :: p(3,4,n), q(3,4,n)
        complex(c_double_complex) :: p_new
        integer :: i, j, channel, k
        do j = 1, 4
            do i = 1, 3
                channel = 3*(j-1) + i
                fft_input(:,channel) = p(i,j,:)
                fft_input(:,channel+12) = q(i,j,:)
            end do
        end do
        call fftw_execute_dft_r2c(forward_plan, fft_input, fft_output)
        do j = 1, 4
            do i = 1, 3
                channel = 3*(j-1) + i
                do k = 1, n/2+1
                    p_new = coefficients(1,k,j)*fft_output(k,channel) + &
                        coefficients(2,k,j)*fft_output(k,channel+12)
                    fft_output(k,channel+12) = coefficients(3,k,j)*fft_output(k,channel) + &
                        coefficients(4,k,j)*fft_output(k,channel+12)
                    fft_output(k,channel) = p_new
                end do
            end do
        end do
        call fftw_execute_dft_c2r(backward_plan, fft_output, fft_input)
        do j = 1, 4
            do i = 1, 3
                channel = 3*(j-1) + i
                p(i,j,:) = fft_input(:,channel) / n
                q(i,j,:) = fft_input(:,channel+12) / n
            end do
        end do
    end subroutine fftw_cayley

    subroutine single_step(n, method, p, q) bind(c)
        use dense_repository_reference, only: dense_repository_cayley
        integer(c_int), value :: n, method
        real(c_double), intent(inout) :: p(3,4,n), q(3,4,n)
        select case(method)
        case(0)
            call dense_repository_cayley(p, q, 4, n)
        case(1)
            call dense_cached(p, q, n)
        case(2)
            call fftw_cayley(p, q, n)
        case(3)
            call free_ring_polymer_cayley_step(p, q, 4, n)
        case default
            stop 5
        end select
    end subroutine single_step

    subroutine advance(n, method, reps, p, q) bind(c)
        integer(c_int), value :: n, method, reps
        real(c_double), intent(inout) :: p(3,4,n), q(3,4,n)
        integer :: k
        do k = 1, reps
            call single_step(n, method, p, q)
        end do
    end subroutine advance

    subroutine measure(n, method, with_pes, reps, p0, q0, p, q, wall, cpu) bind(c)
        integer(c_int), value :: n, method, with_pes, reps
        real(c_double), intent(in) :: p0(3,4,n), q0(3,4,n)
        real(c_double), intent(out) :: p(3,4,n), q(3,4,n), wall, cpu
        real(c_double) :: values(n), g0(3,4,n), g(3,4,n), cpu0, cpu1
        integer(c_int64_t) :: t0, t1, rate
        integer :: k, status
        if (with_pes == 1) then
            call get_potential(q0, values, g0, 4, n, status)
            if (status /= 0) stop 6
        end if
        p = p0
        q = q0
        call cpu_time(cpu0)
        call system_clock(t0, rate)
        do k = 1, reps
            if (with_pes == 1) then
                p = p0 - 0.5d0 * dt * g0
                q = q0
            end if
            call single_step(n, method, p, q)
            if (with_pes == 1) then
                call get_potential(q, values, g, 4, n, status)
                if (status /= 0) stop 7
                p = p - 0.5d0 * dt * g
            end if
        end do
        call system_clock(t1)
        call cpu_time(cpu1)
        wall = dble(t1-t0) / dble(rate) / reps
        cpu = (cpu1-cpu0) / reps
    end subroutine measure

end module benchmark_backends
