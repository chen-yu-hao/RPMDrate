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

    ! Allow an RPMD trajectory to equilibrate in the presence of an Andersen
    ! thermostat, with option to constrain the trajectory to the transition
    ! state dividing surface.
    ! Parameters:
    !   t - The initial time
    !   p - The initial momentum of each bead in each atom
    !   q - The initial position of each bead in each atom
    !   Natoms - The number of atoms in the molecular system
    !   Nbeads - The number of beads to use per atom
    !   steps - The number of time steps to take in this trajectory
    !   xi_current - The current centroid value of the reaction coordinate
    !   potential - A function that evaluates the potential and force for a given position
    !   kforce - The umbrella potential force constant
    !   constrain - 1 to constrain to dividing surface, 0 otherwise
    !   save_trajectory - 1 to save the trajectory to disk for visualization (slow!), 0 otherwise
    ! Returns:
    !   result - 0 if the trajectory evolution was successful, nonzero if unsuccessful
    subroutine equilibrate(t, p, q, Natoms, Nbeads, steps, &
        xi_current, potential, kforce, constrain, save_trajectory, result,strxi,NumWindow,everycount)

        use transition_state, only: check_for_valid_position

        implicit none
        external potential
        integer, intent(in) :: Natoms, Nbeads,NumWindow,everycount
        character(len=*) :: strxi
        double precision, intent(inout) :: t, p(3,Natoms,Nbeads), q(3,Natoms,Nbeads)
        integer, intent(in) :: steps
        double precision, intent(in) :: xi_current, kforce
        integer, intent(in) :: constrain, save_trajectory
        integer, intent(out) :: result
        double precision :: V(Nbeads), dVdq(3,Natoms,Nbeads)
        double precision :: xi, dxi(3,Natoms), d2xi(3,Natoms,3,Natoms)
        double precision :: centroid(3,Natoms)
        integer :: step, andersen_sampling_steps

        result = 0

        ! Set up Andersen thermostat (if turned on)
        andersen_sampling_steps = int(andersen_sampling_time / dt)
        if (thermostat .eq. 1) then
            if (andersen_sampling_time .gt. 0) then
                andersen_sampling_steps = int(andersen_sampling_time / dt)
            else
                andersen_sampling_steps = int(dsqrt(dble(steps)))
            end if
        end if
        ! Set up GLE thermostat (if turned on)
        if (thermostat .eq. 2) then
            call gle_initialize(dt, Natoms, Nbeads, gle_A(1:gle_Ns+1,1:gle_Ns+1), gle_C(1:gle_Ns+1,1:gle_Ns+1), gle_Ns)
        end if

        if (save_trajectory .eq. 1) then
            open(unit=4*NumWindow+900,file='tra/eq_beads/equilibrate_'//strxi//".xyz")
            open(unit=4*NumWindow+901,file='tra/eq_centroid/equilibrate_centroid_'//strxi//".xyz")
            open(unit=4*NumWindow+902,file='tra/eq_xi/equilibrate_xi_'//strxi//".dat")
        end if

        call get_centroid(q, Natoms, Nbeads, centroid)
        call get_reaction_coordinate(centroid, Natoms, xi_current, xi, dxi, d2xi)
        call potential(q, V, dVdq, Natoms, Nbeads, result)
        if (result > 0) then
            ! The initial position is unphysical, so abort
            result = -1
            return
        end if
        if (mode .eq. 1) then
            call add_umbrella_potential(xi, dxi, V, dVdq, Natoms, Nbeads, xi_current, kforce)
            call add_bias_potential(dxi, d2xi, V, dVdq, Natoms, Nbeads)
        end if

        ! Apply GLE thermostat if turned on
        if (thermostat .eq. 2) then
            call gle_thermostat(p, mass, beta, dxi, Natoms, Nbeads, gle_Ns, constrain, result)
            if (result .ne. 0) return
        end if

        do step = 1, steps
            call verlet_step(t, p, q, V, dVdq, xi, dxi, d2xi, Natoms, Nbeads, &
                xi_current, potential, kforce, constrain, result)
            if (result .ne. 0) exit

            ! If constraining to dividing surface, check that the values of
            ! the forming and breaking bonds are reasonable
            if (constrain .eq. 1) then
                call get_centroid(q, Natoms, Nbeads, centroid)
                call check_for_valid_position(centroid, Natoms, 20.0d0, result)
                if (result .ne. 0) then
                    write (*,fmt='(A)') &
                        'Error: Invalid geometry for recrossing factor parent trajectory. Restarting trajectory.'
                    exit
                end if
            end if
            ! write 
            if (save_trajectory .eq. 1) then
                if (mod(step,everycount) .eq. 1) then
                    call get_centroid(q, Natoms, Nbeads, centroid)
                    ! call get_reaction_coordinate(centroid, Natoms, xi_current, xi_new, dxi_new, d2xi_new)
                    call update_vmd_output(q, Natoms, Nbeads, xi ,NumWindow*4+900, NumWindow*4+901,NumWindow*4+902)
                end if
            end if 
            ! Apply Andersen thermostat (if turned on)
            if (thermostat .eq. 1) then
                if (mod(step, andersen_sampling_steps) .eq. 0) call sample_momentum(p, mass, beta, Natoms, Nbeads)
            end if
            ! Apply GLE thermostat if turned on
            if (thermostat .eq. 2) then
                call gle_thermostat(p, mass, beta, dxi, Natoms, Nbeads, gle_Ns, constrain, result)
                if (result .ne. 0) return
            end if

        end do

        ! Clean up GLE thermostat (if turned on)
        if (thermostat .eq. 2) then
            call gle_cleanup()
        end if

        if (save_trajectory .eq. 1) then
            close(unit=4*NumWindow+900)
            close(unit=4*NumWindow+901)
            close(unit=4*NumWindow+902)
        end if

    end subroutine equilibrate

    ! Conduct a simulation of a RPMD trajectory to update the value of the
    ! recrossing factor.
    ! Parameters:
    !   t - The initial time
    !   p - The initial momentum of each bead in each atom
    !   q - The initial position of each bead in each atom
    !   Natoms - The number of atoms in the molecular system
    !   Nbeads - The number of beads to use per atom
    !   steps - The number of time steps to take in this trajectory
    !   xi_current - The current centroid value of the reaction coordinate
    !   potential - A function that evaluates the potential and force for a given position
    !   save_trajectory - 1 to save the trajectory to disk for visualization (slow!), 0 otherwise
    !   kappa_num - The numerator of the recrossing factor expression
    !   kappa_denom - The denominator of the recrossing factor expression
    ! Returns:
    !   result - 0 if the trajectory evolution was successful, nonzero if unsuccessful
    subroutine recrossing_trajectory(t, p, q, Natoms, Nbeads, steps, &
        xi_current, potential, save_trajectory, kappa_num, kappa_denom, result)

        implicit none

        external potential
        integer, intent(in) :: Natoms, Nbeads
        double precision, intent(inout) :: t, p(3,Natoms,Nbeads), q(3,Natoms,Nbeads)
        integer, intent(in) :: steps
        double precision, intent(in) :: xi_current
        double precision, intent(inout) :: kappa_num(steps), kappa_denom
        integer, intent(in) :: save_trajectory
        integer, intent(out) :: result

        double precision :: V(Nbeads), dVdq(3,Natoms,Nbeads)
        double precision :: xi, dxi(3,Natoms), d2xi(3,Natoms,3,Natoms)
        double precision :: centroid(3,Natoms), vs, fs
        integer :: step

        result = 0

        ! if (save_trajectory .eq. 1) then
        !     open(unit=777,file='child.xyz')
        !     open(unit=888,file='child_centroid.xyz')
        ! end if

        call get_centroid(q, Natoms, Nbeads, centroid)
        call get_reaction_coordinate(centroid, Natoms, xi_current, xi, dxi, d2xi)
        call potential(q, V, dVdq, Natoms, Nbeads, result)
        if (result > 0) then
            ! The initial position is unphysical, so abort
            result = -1
            return
        end if
 
        call get_recrossing_velocity(p, dxi, Natoms, Nbeads, vs)
        call get_recrossing_flux(dxi, Natoms, fs)
        if (vs .gt. 0) kappa_denom = kappa_denom + vs / fs

        do step = 1, steps
            call verlet_step(t, p, q, V, dVdq, xi, dxi, d2xi, Natoms, Nbeads, &
                xi_current, potential, 0.d0, 0, result)
            if (result .ne. 0) exit
            ! if (save_trajectory .eq. 1) then
            ! if (mod(step,5000) .eq. 1) then 
            !     call update_vmd_output(q, Natoms, Nbeads, 777, 888)
            if (xi .gt. 0) kappa_num(step) = kappa_num(step) + vs / fs
        end do

        ! if (save_trajectory .eq. 1) then
        !     close(unit=777)
        !     close(unit=888)
        ! end if

    end subroutine recrossing_trajectory

    subroutine umbrella_trajectory(t, p, q, Natoms, Nbeads, steps, &
        xi_current, potential, kforce, xi_range, save_trajectory, av, av2, &
        actual_steps, result,strxi,NumWindow,everycount)

        use transition_state, only: check_for_valid_position, check_values

        implicit none
            external potential
            integer, intent(in) :: Natoms, Nbeads
            integer, intent(in) :: NumWindow,everycount
            character(len=*) :: strxi
            double precision, intent(inout) :: t, p(3,Natoms,Nbeads), q(3,Natoms,Nbeads)
            double precision, intent(in) :: xi_current, kforce, xi_range
            integer, intent(in) :: steps
            integer, intent(in) :: save_trajectory
            double precision, intent(out) :: av, av2
            integer, intent(out) :: actual_steps, result
            double precision :: xi_new, dxi_new, d2xi_new
            double precision :: V(Nbeads), dVdq(3,Natoms,Nbeads)
            double precision :: xi, dxi(3,Natoms), d2xi(3,Natoms,3,Natoms)
            double precision :: centroid(3,Natoms)
            integer :: step, andersen_sampling_steps

        result = 0
        actual_steps = 0

        av = 0.0d0
        av2 = 0.0d0

        ! Set up Andersen thermostat (if turned on)
        andersen_sampling_steps = int(andersen_sampling_time / dt)
        if (thermostat .eq. 1) then
            if (andersen_sampling_time .gt. 0) then
                andersen_sampling_steps = floor(andersen_sampling_time / dt)
            else
                andersen_sampling_steps = floor(dsqrt(dble(steps)))
            end if
        end if
        ! Set up GLE thermostat (if turned on)
        if (thermostat .eq. 2) then
            call gle_initialize(dt, Natoms, Nbeads, gle_A(1:gle_Ns+1,1:gle_Ns+1), gle_C(1:gle_Ns+1,1:gle_Ns+1), gle_Ns)
        end if
        ! write open file to write trajectory
        if (save_trajectory .eq. 1) then
            open(unit=NumWindow*4+10,file='tra/child_beads/child_'//strxi//".xyz")
            open(unit=NumWindow*4+11,file='tra/child_centroid/child_centroid_'//strxi//".xyz")
            open(unit=NumWindow*4+12,file='tra/child_xi/child_xi_'//strxi//".dat")
        end if

        call get_centroid(q, Natoms, Nbeads, centroid)
        call get_reaction_coordinate(centroid, Natoms, xi_current, xi, dxi, d2xi)
        call potential(q, V, dVdq, Natoms, Nbeads, result)
        if (result > 0) then
            ! The initial position is unphysical, so abort
            result = -1
            return
        end if
        call add_umbrella_potential(xi, dxi, V, dVdq, Natoms, Nbeads, xi_current, kforce)
        call add_bias_potential(dxi, d2xi, V, dVdq, Natoms, Nbeads)

        ! Apply GLE thermostat if turned on
        if (thermostat .eq. 2) then
            call gle_thermostat(p, mass, beta, dxi, Natoms, Nbeads, gle_Ns, 0, result)
            if (result .ne. 0) return
        end if

        do step = 1, steps

            call verlet_step(t, p, q, V, dVdq, xi, dxi, d2xi, Natoms, Nbeads, &
                xi_current, potential, kforce, 0, result)
            if (xi_range .ne. 0.0d0 .and. abs(xi - xi_current) > xi_range) then
                actual_steps = step - 1
                exit
            end if
            if (result .ne. 0) then
                actual_steps = step - 1
                exit
            end if

            ! Check that the values of the forming and breaking bonds are reasonable
            call get_centroid(q, Natoms, Nbeads, centroid)
            call check_for_valid_position(centroid, Natoms, 200.0d0, result)
            call check_values(centroid, Natoms, result)
            if (result .ne. 0) then
                write (*,fmt='(A)') &
                    'Error: Invalid geometry for umbrella sampling trajectory. Restarting trajectory.'
                actual_steps = step - 1
                exit
            end if

            if (save_trajectory .eq. 1) then
                if (mod(step,everycount) .eq. 1) then
                    call get_centroid(q, Natoms, Nbeads, centroid)
                    ! call get_reaction_coordinate(centroid, Natoms, xi_current, xi_new, dxi_new, d2xi_new)
                    call update_vmd_output(q, Natoms, Nbeads, xi ,NumWindow*4+10, NumWindow*4+11,NumWindow*4+12)
                end if
            end if
            av = av + xi
            av2 = av2 + xi * xi

            ! Apply Andersen thermostat (if turned on)
            if (thermostat .eq. 1) then
                if (mod(step, andersen_sampling_steps) .eq. 0) call sample_momentum(p, mass, beta, Natoms, Nbeads)
            end if
            ! Apply GLE thermostat if turned on
            if (thermostat .eq. 2) then
                call gle_thermostat(p, mass, beta, dxi, Natoms, Nbeads, gle_Ns, 0, result)
                if (result .ne. 0) return
            end if

        end do

        actual_steps = steps

        ! Clean up GLE thermostat (if turned on)
        if (thermostat .eq. 2) then
            call gle_cleanup()
        end if

        if (save_trajectory .eq. 1) then
            close(unit=4*NumWindow+10)
            close(unit=4*NumWindow+11)
            close(unit=4*NumWindow+12)
        end if

    end subroutine umbrella_trajectory

    ! Advance the simluation by one time step using the velocity Verlet
    ! algorithm.
    ! Parameters:
    !   t - The current simulation time
    !   p - The momentum of each bead in each atom
    !   q - The position of each bead in each atom
    !   V - The potential of each bead
    !   dVdq - The force exerted on each bead in each atom
    !   xi - The value of the reaction coordinate
    !   dxi - The gradient of the reaction coordinate
    !   d2xi - The Hessian of the reaction coordinate
    !   Natoms - The number of atoms in the molecular system
    !   Nbeads - The number of beads to use per atom
    !   xi_current - The current centroid value of the reaction coordinate
    !   potential - A function that evaluates the potential and force for a given position
    !   kforce - The umbrella potential force constant
    !   constrain - 1 to constrain to the transition state dividing surface, 0 otherwise
    ! Returns:
    !   t - The updated simulation time
    !   p - The updated momentum of each bead in each atom
    !   q - The updated position of each bead in each atom
    !   V - The updated potential of each bead
    !   dVdq - The updated force exerted on each bead in each atom
    !   xi - The updated value of the reaction coordinate
    !   dxi - The updated gradient of the reaction coordinate
    !   d2xi - The updated Hessian of the reaction coordinate
    !   result - A flag that indicates if the time step completed successfully (if zero) or that an error occurred (if nonzero)
    subroutine verlet_step(t, p, q, V, dVdq, xi, dxi, d2xi, Natoms, Nbeads, &
        xi_current, potential, kforce, constrain, result)
        implicit none
        external potential, reactants_surface, transition_state_surface
        integer, intent(in) :: Natoms, Nbeads
        double precision, intent(inout) :: t, p(3,Natoms,Nbeads), q(3,Natoms,Nbeads)
        double precision, intent(inout) :: V(Nbeads), dVdq(3,Natoms,Nbeads)
        double precision, intent(inout) :: xi, dxi(3,Natoms), d2xi(3,Natoms,3,Natoms)
        double precision, intent(in) :: xi_current, kforce
        integer, intent(in) :: constrain
        integer, intent(out) :: result

        double precision :: centroid(3,Natoms)
        integer :: i, j

        result = 0

        ! Update momentum (half time step)
        p = p - 0.5d0 * dt * dVdq

        ! Update position (full time step)
        if (Nbeads .eq. 1) then
            ! For a single bead, there are no free ring polymer terms to add,
            ! so we simply update the positions using the momentum, as in
            ! classical trajectories
            do i = 1, 3
                do j = 1, Natoms
                    q(i,j,1) = q(i,j,1) + p(i,j,1) * dt / mass(j)
                end do
            end do
        else
            ! For multiple beads, we update the positions and momenta for the
            ! harmonic free ring term in the Hamiltonian by transforming to
            ! and from normal mode space
            call free_ring_polymer_step(p, q, Natoms, Nbeads)
        end if

        ! If constrain is on, the evolution will be constrained to the
        ! transition state dividing surface
        if (constrain .eq. 1) call constrain_to_dividing_surface(p, q, dxi, Natoms, Nbeads, xi_current, result)
        if (result .ne. 0) return

        ! Update reaction coordinate value, gradient, and Hessian
        call get_centroid(q, Natoms, Nbeads, centroid)
        call get_reaction_coordinate(centroid, Natoms, xi_current, xi, dxi, d2xi)

        ! Update potential and forces using new position
        call potential(q, V, dVdq, Natoms, Nbeads, result)
        if (result > 0) return
        if (mode .eq. 1) then
            call add_umbrella_potential(xi, dxi, V, dVdq, Natoms, Nbeads, xi_current, kforce)
            call add_bias_potential(dxi, d2xi, V, dVdq, Natoms, Nbeads)
        end if

        ! Update momentum (half time step)
        p = p - 0.5d0 * dt * dVdq

        ! Constrain momentum again
        if (constrain .eq. 1) call constrain_momentum_to_dividing_surface(p, dxi, Natoms, Nbeads)

        ! Update time
        t = t + dt

    end subroutine verlet_step
    
subroutine init_dft_CC(Nbeads,dftCC,idftCC)
        implicit none
        integer, intent(in) :: Nbeads
        double precision, intent(in) :: dftCC(Nbeads,Nbeads)
        double precision, intent(in) :: idftCC(Nbeads,Nbeads)
        if (ALLCC==0) then
          allocate(CC(Nbeads,Nbeads))
          allocate(ICC(Nbeads,Nbeads))
          CC=dftCC
          ICC=idftCC
          ALLCC=1
        end if
        ! write(*,*)CC
      end subroutine init_dft_CC
    ! Update the positions and momenta of each atom in each free ring polymer
    ! bead for the term in the Hamiltonian describing the harmonic free ring
    ! polymer interactions. This is most efficiently done in normal mode space;
    ! this function therefore uses fast Fourier transforms (from the FFTW3
    ! library) to transform to and from normal mode space.
    ! Parameters:
    !   p - The momentum of each bead in each atom
    !   q - The position of each bead in each atom
    !   Natoms - The number of atoms in the molecular system
    !   Nbeads - The number of beads to use per atom
    ! Returns:
    !   p - The updated momentum of each bead in each atom
    !   q - The updated position of each bead in each atom
    subroutine free_ring_polymer_step(p, q, Natoms, Nbeads)

        implicit none
        integer, intent(in) :: Natoms, Nbeads
        double precision, intent(inout) :: p(3,Natoms,Nbeads), q(3,Natoms,Nbeads)

        double precision :: poly(4,Nbeads)
        double precision :: beta_n, twown, pi_n, wk, wt, wm, cos_wt, sin_wt, p_new
        integer :: i, j, k

        ! Transform to normal mode space
        do i = 1, 3
            do j = 1, Natoms
                p(i,j,:)=matmul (CC,p(i,j,:))
                q(i,j,:)=matmul (CC,q(i,j,:))
            end do
        end do

        do j = 1, Natoms

            poly(1,1) = 1.0d0
            poly(2,1) = 0.0d0
            poly(3,1) = dt / mass(j)
            poly(4,1) = 1.0d0

            if (Nbeads .gt. 1) then
                beta_n = beta / Nbeads
                twown = 2.0d0 / beta_n
                pi_n = pi / Nbeads
                do k = 1, Nbeads / 2
                    wk = twown * dsin(k * pi_n)
                    wt = wk * dt
                    wm = wk * mass(j)
                    cos_wt = dcos(wt)
                    sin_wt = dsin(wt)
                    poly(1,k+1) = cos_wt
                    poly(2,k+1) = -wm*sin_wt
                    poly(3,k+1) = sin_wt/wm
                    poly(4,k+1) = cos_wt
                end do
                do k = 1, (Nbeads - 1) / 2
                    poly(1,Nbeads-k+1) = poly(1,k+1)
                    poly(2,Nbeads-k+1) = poly(2,k+1)
                    poly(3,Nbeads-k+1) = poly(3,k+1)
                    poly(4,Nbeads-k+1) = poly(4,k+1)
                end do
            end if

            do k = 1, Nbeads
                do i = 1, 3
                    p_new = p(i,j,k) * poly(1,k) + q(i,j,k) * poly(2,k)
                    q(i,j,k) = p(i,j,k) * poly(3,k) + q(i,j,k) * poly(4,k)
                    p(i,j,k) = p_new
                end do
            end do

        end do

        ! Transform back to Cartesian space
        do i = 1, 3
            do j = 1, Natoms
                p(i,j,:)=matmul (ICC,p(i,j,:))
                q(i,j,:)=matmul (ICC,q(i,j,:))
            end do
        end do

    end subroutine free_ring_polymer_step

    ! Constrain the position and the momentum to the dividing surface, using the
    ! SHAKE/RATTLE algorithm.
    ! Parameters:
    !   p - The momentum of each bead in each atom
    !   q - The position of each bead in each atom
    !   dxi - The gradient of the reaction coordinate
    !   Natoms - The number of atoms in the molecular system
    !   Nbeads - The number of beads to use per atom
    ! Returns:
    !   p - The constrained momentum of each bead in each atom
    !   q - The constrained position of each bead in each atom
    !   info - 0 if the constraining was successful, 1 if unsuccessful
    subroutine constrain_to_dividing_surface(p, q, dxi, Natoms, Nbeads, xi_current, info)

        implicit none
        integer, intent(in) :: Natoms, Nbeads
        double precision, intent(inout) :: p(3,Natoms,Nbeads), q(3,Natoms,Nbeads)
        double precision, intent(inout) :: dxi(3,Natoms)
        double precision, intent(in) :: xi_current

        double precision :: centroid(3,Natoms), qctemp(3,Natoms)
        integer :: i, j, k, maxiter, iter
        double precision :: xi_new, dxi_new(3,Natoms), d2xi_new(3,Natoms,3,Natoms)
        double precision :: mult, sigma, dsigma, dx, coeff
        integer, intent(out) :: info

        call get_centroid(q, Natoms, Nbeads, centroid)

        ! The Lagrange multiplier for the constraint
        mult = 0.0d0

        qctemp(:,:) = 0.0d0

        info = 0

        maxiter = 100
        do iter = 1, maxiter

            coeff = mult * dt * dt / Nbeads

            do i = 1, 3
                do j = 1, Natoms
                    qctemp(i,j) = centroid(i,j) + coeff * dxi(i,j) / mass(j)
                end do
            end do

            call get_reaction_coordinate(qctemp, Natoms, xi_current, xi_new, dxi_new, d2xi_new)

            sigma = xi_new
            dsigma = 0.0d0
            do i = 1, 3
                do j = 1, Natoms
                    dsigma = dsigma + dxi_new(i,j) * dt * dt * dxi(i,j) / (mass(j) * Nbeads)
                end do
            end do

            dx = sigma / dsigma
            mult = mult - dx
            if (dabs(dx) .lt. 1.0d-8 .or. dabs(sigma) .lt. 1.0d-10) exit

            if (iter .eq. maxiter) then
                write (*,fmt='(A)') 'Error: SHAKE exceeded maximum number of iterations. Restarting trajectory.'
                write (*,fmt='(A,E13.5,A,E13.5)') 'dx = ', dx, ', sigma = ', sigma
                info = 1
                return
            end if

        end do

        do i = 1, 3
            do j = 1, Natoms
                do k = 1, Nbeads
                    q(i,j,k) = q(i,j,k) + coeff / mass(j) * dxi(i,j)
                    p(i,j,k) = p(i,j,k) + mult * dt / Nbeads * dxi(i,j)
                end do
            end do
        end do

    end subroutine constrain_to_dividing_surface

    ! Constrain the momentum to the reaction coordinate, to ensure that the time
    ! derivative of the dividing surface is zero.
    ! Parameters:
    !   p - The momentum of each bead in each atom
    !   dxi - The gradient of the reaction coordinate
    !   Natoms - The number of atoms in the molecular system
    !   Nbeads - The number of beads to use per atom
    ! Returns:
    !   p - The constrained momentum of each bead in each atom
    subroutine constrain_momentum_to_dividing_surface(p, dxi, Natoms, Nbeads)

        implicit none
        integer, intent(in) :: Natoms, Nbeads
        double precision, intent(in) :: dxi(3,Natoms)
        double precision, intent(inout) :: p(3,Natoms,Nbeads)

        double precision :: coeff1, coeff2, lambda
        integer :: i, j, k

        coeff1 = 0.0d0
        do i = 1, 3
            do j = 1, Natoms
                do k = 1, Nbeads
                    coeff1 = coeff1 + dxi(i,j) * p(i,j,k) / mass(j)
                end do
            end do
        end do

        coeff2 = 0.0d0
        do i = 1, 3
            do j = 1, Natoms
                coeff2 = coeff2 + dxi(i,j) * dxi(i,j) / mass(j)
            end do
        end do

        lambda = -coeff1 / coeff2 / Nbeads
        do i = 1, 3
            do j = 1, Natoms
                do k = 1, Nbeads
                    p(i,j,k) = p(i,j,k) + lambda * dxi(i,j)
                end do
            end do
        end do

        ! DEBUG: Check that constraint is correct: coeff1 should now evaluate to
        ! zero within numerical precision
        !coeff1 = 0.0d0
        !do i = 1, 3
        !    do j = 1, Natoms
        !        do k = 1, Nbeads
        !            coeff1 = coeff1 + dxi(i,j) * p(i,j,k) / mass(j)
        !        end do
        !    end do
        !end do

    end subroutine constrain_momentum_to_dividing_surface

    ! Add an umbrella potential and the corresponding forces to the overall
    ! potential and forces of the RPMD system.
    ! Parameters:
    !   xi - The value of the reaction coordinate
    !   dxi - The gradient of the reaction coordinate
    !   V - The potential of each bead
    !   dVdq - The force exerted on each bead in each atom
    !   Natoms - The number of atoms in the molecular system
    !   Nbeads - The number of beads to use per atom
    !   xi_current - The current centroid value of the reaction coordinate
    !   kforce - The umbrella potential force constant
    ! Returns:
    !   V - The updated potential of each bead
    !   dVdq - The updated force exerted on each bead in each atom
    subroutine add_umbrella_potential(xi, dxi, V, dVdq, Natoms, Nbeads, xi_current, kforce)

        implicit none
        external potential
        integer, intent(in) :: Natoms, Nbeads
        double precision, intent(in) :: xi, dxi(3,Natoms)
        double precision, intent(in) :: xi_current, kforce
        double precision, intent(inout) :: V(Nbeads), dVdq(3,Natoms,Nbeads)

        double precision :: delta
        integer :: i, j, k

        delta = xi - xi_current

        ! Add umbrella potential
        do k = 1, Nbeads
            V(k) = V(k) + 0.5d0 * kforce * delta * delta
        end do

        ! Add umbrella force
        do i = 1, 3
            do j = 1, Natoms
                do k = 1, Nbeads
                    dVdq(i,j,k) = dVdq(i,j,k) + kforce * delta * dxi(i,j)
                end do
            end do
        end do

    end subroutine add_umbrella_potential

    ! Add a bias potential and the corresponding forces to the overall
    ! potential and forces of the RPMD system.
    ! Parameters:
    !   dxi - The gradient of the reaction coordinate
    !   d2xi - The Hessian of the reaction coordinate
    !   V - The potential of each bead
    !   dVdq - The force exerted on each bead in each atom
    !   Natoms - The number of atoms in the molecular system
    !   Nbeads - The number of beads to use per atom
    ! Returns:
    !   V - The updated potential of each bead
    !   dVdq - The updated force exerted on each bead in each atom
    subroutine add_bias_potential(dxi, d2xi, V, dVdq, Natoms, Nbeads)

        implicit none
        integer, intent(in) :: Natoms, Nbeads
        double precision, intent(in) :: dxi(3,Natoms), d2xi(3,Natoms,3,Natoms)
        double precision, intent(inout) :: V(Nbeads), dVdq(3,Natoms,Nbeads)

        double precision :: fs, fs2, log_fs, coeff1, coeff2, dhams
        integer :: i, j, k, i2, j2

        fs2 = 0.0d0
        do i = 1, 3
            do j = 1, Natoms
                fs2 = fs2 + dxi(i,j) * dxi(i,j) / mass(j)
            end do
        end do
        coeff1 = 2.0d0 * pi * beta
        fs2 = fs2 / coeff1
        fs = sqrt(fs2)
        log_fs = log(fs)
        coeff2 = -1.0d0 / beta

        ! Add bias term to potential
        do k = 1, Nbeads
            V(k) = V(k) + coeff2 * log_fs
        end do

        ! Add bias term to forces
        do i = 1, 3
            do j = 1, Natoms
                dhams = 0.0d0
                do i2 = 1, 3
                    do j2 = 1, Natoms
                        dhams = dhams + d2xi(i2,j2,i,j) * dxi(i2,j2) / mass(j2)
                    end do
                end do
                dhams = dhams * coeff2 / (coeff1 * fs2)
                do k = 1, Nbeads
                    dVdq(i,j,k) = dVdq(i,j,k) + dhams
                end do
            end do
        end do

    end subroutine add_bias_potential

    ! Compute the value, gradient, and Hessian of the reaction coordinate.
    ! Parameters:
    !   centroid - The centroid of each atom
    !   Natoms - The number of atoms in the molecular system
    ! Returns:
    !   xi - The value of the reaction coordinate
    !   dxi - The gradient of the reaction coordinate
    !   d2xi - The Hessian of the reaction coordinate
    subroutine get_reaction_coordinate(centroid, Natoms, xi_current, xi, dxi, d2xi)

        use reactants, only: reactants_value => value, &
            reactants_gradient => gradient, &
            reactants_hessian => hessian
        use transition_state, only: transition_state_value => value, &
            transition_state_gradient => gradient, &
            transition_state_hessian => hessian

        implicit none
        integer, intent(in) :: Natoms
        double precision, intent(in) :: centroid(3,Natoms)
        double precision, intent(in) :: xi_current
        double precision, intent(out) :: xi, dxi(3,Natoms), d2xi(3,Natoms,3,Natoms)

        double precision :: s0, ds0(3,Natoms), d2s0(3,Natoms,3,Natoms)
        double precision :: s1, ds1(3,Natoms), d2s1(3,Natoms,3,Natoms)
        integer :: i1, i2, j1, j2

        xi = 0.0d0
        dxi(:,:) = 0.0d0
        d2xi(:,:,:,:) = 0.0d0

        ! Evaluate reactants dividing surface value, gradient, and Hessian
        call reactants_value(centroid, Natoms, s0)
        call reactants_gradient(centroid, Natoms, ds0)
        call reactants_hessian(centroid, Natoms, d2s0)

        ! Evaluate transition state dividing surface value, gradient, and Hessian
        call transition_state_value(centroid, Natoms, s1)
        call transition_state_gradient(centroid, Natoms, ds1)
        call transition_state_hessian(centroid, Natoms, d2s1)

        ! Compute reaction coordinate value, gradient, and Hessian
        ! The functional form is different depending on the type of RPMD
        ! calculation we are performing
        if (mode .eq. 1) then
            ! Umbrella integration
            xi = s0 / (s0 - s1)
            dxi = (s0 * ds1 - s1 * ds0) / ((s0 - s1) * (s0 - s1))
            do i1 = 1, 3
                do j1 = 1, Natoms
                    do i2 = 1, 3
                        do j2 = 1, Natoms
                            d2xi(i1,j1,i2,j2) = ((s0 * d2s1(i1,j1,i2,j2) + ds0(i2,j2) * ds1(i1,j1) &
                                - ds1(i2,j2) * ds0(i1,j1) - s1 * d2s0(i1,j1,i2,j2)) * (s0 - s1) &
                                - 2.0d0 * (s0 * ds1(i1,j1) - s1 * ds0(i1,j1)) &
                                * (ds0(i2,j2) - ds1(i2,j2))) &
                                / ((s0 - s1) * (s0 - s1) * (s0 - s1))
                        end do
                    end do
                end do
            end do
        elseif (mode .eq. 2) then
            ! Recrossing factor
            xi = xi_current * s1 + (1 - xi_current) * s0
            dxi = xi_current * ds1 + (1 - xi_current) * ds0
            d2xi = xi_current * d2s1 + (1 - xi_current) * d2s0
        else
            write (*,fmt='(A,I3,A)') 'Invalid mode ', mode, ' encountered in get_reaction_coordinate().'
            stop
        end if

    end subroutine get_reaction_coordinate

    ! Return the flux used to compute the recrossing factor.
    ! Parameters:
    !   dxi - The gradient of the reaction coordinate
    !   Natoms - The number of atoms in the molecular system
    !   Nbeads - The number of beads to use per atom
    ! Returns:
    !   fs - The flux used to compute the recrossing factor
    subroutine get_recrossing_flux(dxi, Natoms, fs)

        implicit none
        integer, intent(in) :: Natoms
        double precision, intent(in) :: dxi(3,Natoms)
        double precision, intent(out) :: fs

        integer :: i, j

        fs = 0.0d0
        do i = 1, 3
            do j = 1, Natoms
                fs = fs + dxi(i,j) * dxi(i,j) / mass(j)
            end do
        end do
        fs = sqrt(fs / (2.0d0 * pi * beta))

    end subroutine get_recrossing_flux

    ! Return the velocity used to compute the recrossing factor.
    ! Parameters:
    !   p - The momentum of each bead in each atom
    !   dxi - The gradient of the reaction coordinate
    !   Natoms - The number of atoms in the molecular system
    !   Nbeads - The number of beads to use per atom
    ! Returns:
    !   vs - The velocity used to compute the recrossing factor
    subroutine get_recrossing_velocity(p, dxi, Natoms, Nbeads, vs)

        implicit none
        integer, intent(in) :: Natoms, Nbeads
        double precision, intent(in) :: p(3,Natoms,Nbeads), dxi(3,Natoms)
        double precision, intent(out) :: vs

        integer :: i, j, k

        vs = 0.0d0
        do i = 1, 3
            do j = 1, Natoms
                do k = 1, Nbeads
                    vs = vs + dxi(i,j) * p(i,j,k) / mass(j)
                end do
            end do
        end do
        vs = vs / Nbeads

    end subroutine get_recrossing_velocity

    ! Return a pseudo-random sampling of momenta from a Boltzmann distribution at
    ! the temperature of interest.
    ! Parameters:
    !   Natoms - The number of atoms in the molecular system
    !   Nbeads - The number of beads to use per atom
    ! Returns:
    !   p - The sampled momentum of each bead in each atom
    subroutine sample_momentum(p, mass, beta, Natoms, Nbeads)

        implicit none
        integer, intent(in) :: Natoms, Nbeads
        double precision, intent(in) :: mass(Natoms)
        double precision, intent(in) :: beta
        double precision, intent(out) :: p(3,Natoms,Nbeads)

        double precision :: beta_n, dp(Natoms)
        integer :: i, j, k

        beta_n = beta / Nbeads
        dp = sqrt(mass / beta_n)

        do i = 1, 3
            do j = 1, Natoms
                do k = 1, Nbeads
                    call randomn(p(i,j,k))
                    p(i,j,k) = p(i,j,k) * dp(j)
                end do
            end do
        end do

    end subroutine sample_momentum

    ! Compute the total energy of all ring polymers in the RPMD system.
    ! Parameters:
    !   q - The position of each bead in each atom
    !   Natoms - The number of atoms in the molecular system
    !   Nbeads - The number of beads to use per atom
    ! Returns:
    !   Ering - The total energy of all ring polymers
    subroutine get_ring_polymer_energy(q, Natoms, Nbeads, Ering)

        implicit none
        integer, intent(in) :: Natoms, Nbeads
        double precision, intent(in) :: q(3,Natoms,Nbeads)
        double precision, intent(out) :: Ering

        double precision :: wn, dx, dy, dz
        integer :: j, k

        Ering = 0.0d0
        wn = Nbeads / beta
        do j = 1, Natoms
            dx = q(1,j,1) - q(1,j,Nbeads)
            dy = q(2,j,1) - q(2,j,Nbeads)
            dz = q(3,j,1) - q(3,j,Nbeads)
            Ering = Ering + 0.5d0 * mass(j) * wn * wn * (dx * dx + dy * dy + dz * dz)
            do k = 2, Nbeads
                dx = q(1,j,k-1) - q(1,j,k)
                dy = q(2,j,k-1) - q(2,j,k)
                dz = q(3,j,k-1) - q(3,j,k)
                Ering = Ering + 0.5d0 * mass(j) * wn * wn * (dx * dx + dy * dy + dz * dz)
            end do
        end do

    end subroutine get_ring_polymer_energy

    ! Compute the total kinetic energy of the RPMD system.
    ! Parameters:
    !   p - The momentum of each bead in each atom
    !   Natoms - The number of atoms in the molecular system
    !   Nbeads - The number of beads to use per atom
    ! Returns:
    !   Ek - The kinetic energy of the system
    subroutine get_kinetic_energy(p, Natoms, Nbeads, Ek)

        implicit none
        integer, intent(in) :: Natoms, Nbeads
        double precision, intent(in) :: p(3,Natoms,Nbeads)
        double precision, intent(out) :: Ek

        integer :: i, j, k

        Ek = 0.0d0
        do i = 1, 3
            do j = 1, Natoms
                do k = 1, Nbeads
                    Ek = Ek + 0.5d0 * p(i,j,k) * p(i,j,k) / mass(j)
                end do
            end do
        end do

    end subroutine get_kinetic_energy

    ! Compute the center of mass position of the RPMD system.
    ! Parameters:
    !   q - The position of each bead in each atom
    !   Natoms - The number of atoms in the molecular system
    !   Nbeads - The number of beads to use per atom
    ! Returns:
    !   cm - The center of mass of the system
    subroutine get_center_of_mass(q, Natoms, Nbeads, cm)

        implicit none
        integer, intent(in) :: Natoms, Nbeads
        double precision, intent(in) :: q(3,Natoms,Nbeads)
        double precision, intent(out) :: cm(3)

        double precision :: total_mass
        integer :: i, j, k

        cm(:) = 0.0d0
        total_mass = sum(mass)
        do i = 1, 3
            do j = 1, Natoms
                do k = 1, Nbeads
                    cm(i) = cm(i) + q(i,j,k) * mass(j)
                end do
            end do
            cm(i) = cm(i) / total_mass
        end do

    end subroutine get_center_of_mass

    ! Compute the centroid position of each atom in the RPMD system.
    ! Parameters:
    !   q - The position of each bead in each atom
    !   Natoms - The number of atoms in the molecular system
    !   Nbeads - The number of beads to use per atom
    ! Returns:
    !   centroid - The centroid of each atom
    subroutine get_centroid(q, Natoms, Nbeads, centroid)

        implicit none
        integer, intent(in) :: Natoms, Nbeads
        double precision, intent(in) :: q(3,Natoms,Nbeads)
        double precision, intent(out) :: centroid(3,Natoms)

        integer :: i, j, k

        centroid(:,:) = 0.0d0
        do i = 1, 3
            do j = 1, Natoms
                do k = 1, Nbeads
                    centroid(i,j) = centroid(i,j) + q(i,j,k)
                end do
                centroid(i,j) = centroid(i,j) / Nbeads
            end do
        end do

    end subroutine get_centroid

    ! Compute the radius of gyration of each atom in the RPMD system. This is
    ! a useful quantity to check while debugging.
    ! Parameters:
    !   q - The position of each bead in each atom
    !   Natoms - The number of atoms in the molecular system
    !   Nbeads - The number of beads to use per atom
    ! Returns:
    !   R - The radius of gyration of each atom
    subroutine get_radius_of_gyration(q, Natoms, Nbeads, R)

        implicit none
        integer, intent(in) :: Natoms, Nbeads
        double precision, intent(in) :: q(3,Natoms,Nbeads)
        double precision, intent(out) :: R(Natoms)

        double precision :: centroid(3,Natoms), dx
        integer :: i, j, k

        call get_centroid(q, Natoms, Nbeads, centroid)

        R(:) = 0.0d0
        do j = 1, Natoms
            do i = 1, 3
                do k = 1, Nbeads
                    dx = q(i,j,k) - centroid(i,j)
                    R(j) = R(j) + dx * dx
                end do
            end do
            R(j) = sqrt(R(j) / Nbeads)
        end do

    end subroutine get_radius_of_gyration

    ! Write the given position to a pair of VMD output files: one for all beads and
    ! one for the centroid.
    ! Parameters:
    !   q - The position of each bead in each atom
    !   Natoms - The number of atoms in the molecular system
    !   Nbeads - The number of beads to use per atom
    !   beads_file_number - The output file number to save all beads to
    !   centroid_file_number - The output file number to save the centroids to
    subroutine update_vmd_output(q, Natoms, Nbeads, xi, beads_file_number, centroid_file_number,xi_file_number)

        integer, intent(in) :: Natoms, Nbeads
        double precision, intent(in) :: q(3,Natoms,Nbeads)
        double precision, intent(in) :: xi
        integer, intent(in) :: beads_file_number, centroid_file_number, xi_file_number
        integer :: j, k

        double precision :: centroid(3,Natoms)

        call get_centroid(q, Natoms, Nbeads, centroid)
        
        write(beads_file_number,fmt='(I6)') Natoms * Nbeads
        write(beads_file_number,fmt='(A)')
        do j = 1, Natoms
            do k = 1, Nbeads
                write(beads_file_number,fmt='(I4,3F11.6)') j, q(1,j,k), q(2,j,k), q(3,j,k)
            end do
        end do

        write(centroid_file_number,fmt='(I6)') Natoms
        write(centroid_file_number,fmt='(A)')
        do j = 1, Natoms
            write(centroid_file_number,fmt='(I4,3F11.6)') j, centroid(1,j), centroid(2,j), centroid(3,j)
        end do

        write(xi_file_number,fmt='(F11.6)') xi

    end subroutine

        ! Initialize the GLE thermostat by allocating and populating several
    ! temporary arrays.
    subroutine gle_initialize(dt, Natoms, Nbeads, A, C, Ns)

        integer, intent(in) :: Natoms, Nbeads, Ns
        double precision, intent(in) :: dt, A(Ns+1,Ns+1), C(Ns+1,Ns+1)

        double precision :: gr(Ns+1), C1(Ns+1,Ns+1)
        integer :: i, j, k, s

        ! Allocate arrays
        allocate(gle_S(Ns+1,Ns+1))
        allocate(gle_T(Ns+1,Ns+1))
        allocate(gle_p(3,Natoms,Nbeads,Ns+1))
        allocate(gle_np(3,Natoms,Nbeads,Ns+1))

        ! Determine the deterministic part of the propagator
        call matrix_exp(-dt*A, Ns+1, 15, 15, gle_T)

        ! Determine the stochastic part of the propagator
        call cholesky(C - matmul(gle_T, matmul(C, transpose(gle_T))), gle_S, Ns+1)

        ! Initialize the auxiliary noise vectors
        ! To stay general, we use the Cholesky decomposition of C; this allows
        ! for use of non-diagonal C to break detailed balance
        ! We also use an extra slot for the physical momentum, as we could then
        ! use it to initialize the momentum in the calling code
        call cholesky(C, C1, Ns+1)
        do i = 1, 3
            do j = 1, Natoms
                do k = 1, Nbeads
                    do s = 1, Ns+1
                        call randomn(gr(s))
                    end do
                    gle_p(i,j,k,:) = matmul(C1, gr)
                end do
            end do
        end do

    end subroutine gle_initialize

    ! Apply the GLE thermostat to the momentum.
    subroutine gle_thermostat(p, mass, beta, dxi, Natoms, Nbeads, Ns, constrain, result)

        implicit none
        integer, intent(in) :: Natoms, Nbeads, Ns
        double precision, intent(in) :: mass(Natoms), beta, dxi(3,Natoms)
        double precision, intent(inout) :: p(3,Natoms,Nbeads)
        integer, intent(in) :: constrain
        integer, intent(inout) :: result

        double precision :: p0(3,Natoms,Nbeads), p00(3,Natoms,Nbeads), cgj
        integer :: i, j, k, s
        integer :: N, nmrep

        nmrep = 0

        N = 3 * Natoms * Nbeads

        if (constrain .eq. 1) then
            call sample_momentum(p0, mass, beta, Natoms, Nbeads)
            p00(:,:,:) = p0(:,:,:)
            call constrain_momentum_to_dividing_surface(p00, dxi, Natoms, Nbeads)
            p = p + p0 - p00
        end if

        ! Switch to mass-scaled coordinates when storing momenta in gle_p
        do j = 1, Natoms
            gle_p(:,j,:,1) = p(:,j,:) / dsqrt(mass(j))
        end do

        ! We pretend that gp is a (3*Natoms*Nbeads)x(Ns+1) matrix, which should be fine...
        call dgemm('N','T', N, Ns+1, Ns+1, 1.0d0, gle_p, N, gle_T, Ns+1, 0.0d0, gle_np, N)

        ! Compute the random part
        do s = 1, Ns+1
            do i = 1, 3
                do k = 1, Nbeads
                    cgj = 1.0d0     ! This is to make it work when applied in NM representation
                    if (nmrep .gt. 0 .and. k .ne. 1 .and. (mod(Nbeads,2) .ne. 0 .or. k .ne. (Nbeads/2+1))) then
                        cgj = dsqrt(0.5d0)
                    end if
                    do j = 1, Natoms
                        call randomn(gle_p(i,j,k,s))
                        gle_p(i,j,k,s) = gle_p(i,j,k,s) * cgj
                    end do
                end do
            end do
        end do

        ! Again we pretend that gp is a (3*Natoms*Nbeads)x(Ns+1) matrix, which should be fine...
        call dgemm('N','T', N, Ns+1, Ns+1, 1.0d0, gle_p, N, gle_S, Ns+1, 1.0d0, gle_np, N)
        gle_p = gle_np

        ! Switch back from mass-scaled coordinates when recovering momenta from gle_p
        do j = 1, Natoms
            p(:,j,:) = gle_p(:,j,:,1) * dsqrt(mass(j))
        end do

        ! If desired, constrain the momentum to the dividing surface
        if (constrain .eq. 1) call constrain_momentum_to_dividing_surface(p, dxi, Natoms, Nbeads)

    end subroutine gle_thermostat

    ! Clean up the GLE thermostat by deallocating temporary arrays.
    subroutine gle_cleanup()

        deallocate(gle_S, gle_T, gle_p, gle_np)

    end subroutine gle_cleanup

end module system
