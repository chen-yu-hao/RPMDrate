module example
  use iso_c_binding
  implicit none
contains
  subroutine outer(q, atoms, beads)
    integer, intent(in) :: atoms, beads
    real(c_double), target, intent(inout) :: q(3,atoms,beads)
    call inner(q, 3*atoms, beads, c_loc(q(1,1,1)))
  end subroutine
  subroutine inner(q, nd, beads, address)
    integer, intent(in) :: nd, beads
    type(c_ptr), intent(in) :: address
    real(c_double), target, intent(inout) :: q(nd,beads)
    if (.not.c_associated(c_loc(q(1,1)), address)) stop 1
    q(nd,beads) = 12.5d0
  end subroutine
end module
program check
  use example
  real(c_double) :: q(3,4,32)
  q = 0.d0
  call outer(q, 4, 32)
  if (q(3,4,32) /= 12.5d0) stop 2
  print *, 'sequence association: same address, shared update'
end program
