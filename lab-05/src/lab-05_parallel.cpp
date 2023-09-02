#include <deal.II/base/convergence_table.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "Poisson3D_parallel.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  int fd = ::open("/dev/null", O_WRONLY);
  ::dup2(fd, 2);
  ::close(fd);
  // This object calls MPI_Init when it is constructed, and MPI_Finalize when it
  // is destroyed. It also initializes several other libraries bundled with
  // dealii (e.g. p4est, PETSc, ...).
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
  
  dealii::ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  const std::vector<unsigned int> N_values = {4, 9, 19, 39};
  const unsigned int degree = 1;

  // for (const unsigned int &N : N_values){
  unsigned int N = 39;
  Poisson3DParallel problem(N, degree);
  std::ofstream myfile;
  myfile.open ("timer" + std::to_string(N) + ".txt");
  dealii::TimerOutput timer (MPI_COMM_WORLD,
                   pcout,
                   TimerOutput::summary,
                   TimerOutput::wall_times);
  
  timer.enter_subsection ("Setup dof system");
  problem.setup();
  timer.leave_subsection();
  timer.enter_subsection ("Assemble");
  problem.assemble();
  timer.leave_subsection();
  timer.enter_subsection ("Solve");
  problem.solve();
  timer.leave_subsection();
  problem.output();
  myfile.close();
  // }
  return 0;
}