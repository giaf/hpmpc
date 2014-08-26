HPMPC -- Library for High-Performance implementation of solvers for MPC.

The library contains an interior-point (IP) solver for the linear MPC (LMPC) problem with box constraints, and a solver for the the linear-quadratic control problem (LQCP), that is a used as a routine in the IP method. The library is self-contained and also contains the code for the linear-algebra routines.

The code is highly-optimized for a number of common architectures, plus a reference version in plain C code. The architecture can be set in the Makefile.rule file. The code is intended to be used in a Linux machine and using gcc as compiler. Some optimized routine may not work with other compilers.

The folder test_problems contains some test problem for the linear-algebra, for the LQCP solver and for the LMPC solver. The test problem can be chosen by editing the file 

/test_problems/Makefile

The the code comes as a library, that can solver problems of every size. It is generated typing in a terminal the command

$ make

that also runs the test problem.

A code-generated version is currently under development. More documentation will be available soon.

Questions and comments can be send to the author Gianluca Frison, at the email address

giaf (at) dtu.dk


