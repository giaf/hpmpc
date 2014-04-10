###################################################################################################
#                                                                                                 #
#	This file is part of HPMPC.                                                                   #
#                                                                                                 #
#	HPMPC -- Library for High-Performance implementation of solvers for MPC.                      #
#	Copyright (C) 2014 by Technical Univeristy of Denmark. All rights reserved.                   #
#                                                                                                 #
#	HPMPC is free software; you can redistribute it and/or                                        #
#	modify it under the terms of the GNU Lesser General Public                                    #
#	License as published by the Free Software Foundation; either                                  #
#	version 2.1 of the License, or (at your option) any later version.                            #
#                                                                                                 #
#	HPMPC is distributed in the hope that it will be useful,                                      #
#	but WITHOUT ANY WARRANTY; without even the implied warranty of                                #
#	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                                          #
#	See the GNU Lesser General Public License for more details.                                   #
#                                                                                                 #
#	You should have received a copy of the GNU Lesser General Public                              #
#	License along with HPMPC; if not, write to the Free Software                                  #
#	Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA                #
#                                                                                                 #
#	Author: Gianluca Frison, giaf (at) dtu.dk                                                     #
#                                                                                                 #
###################################################################################################


include ./Makefile.rule

ifeq ($(TARGET), AVX)
AUX_OBJS = ./auxiliary/aux_d_c99.o ./auxiliary/aux_s_c99.o ./auxiliary/block_size_avx.o 
KERNEL_OBJS_DOUBLE = ./kernel/kernel_dgemm_avx_lib4.o ./kernel/kernel_dpotrf_sse_lib4.o ./kernel/kernel_dgemv_avx_lib4.o ./kernel/corner_dtrmm_avx_lib4.o ./kernel/corner_dpotrf_sse_lib4.o
BLAS_OBJS = ./blas/blas_d_avx_lib4.o
LQCP_OBJS = ./lqcp_solvers/dricposv.o
LQCP_CODEGEN_OBJS = ./codegen/dricposv_codegen.o
MPC_OBJS = ./mpc_solvers/ip_d_box.o
CFLAGS = $(OPT) -std=c99 -mavx -DTARGET_AVX $(DEBUG)
endif
ifeq ($(TARGET), NEON)
AUX_OBJS = ./auxiliary/aux_d_c99.o ./auxiliary/aux_s_c99.o ./auxiliary/block_size_neon.o 
KERNEL_OBJS_DOUBLE = ./kernel/kernel_dgemm_neon_lib4.o ./kernel/kernel_dpotrf_c99_lib4.o ./kernel/kernel_dgemv_c99_lib4.o ./kernel/corner_dtrmm_c99_lib4.o ./kernel/corner_dpotrf_c99_lib4.o
BLAS_OBJS = ./blas/blas_d_neon_lib4.o
LQCP_OBJS = ./lqcp_solvers/dricposv.o
LQCP_CODEGEN_OBJS = ./codegen/dricposv_codegen.o
MPC_OBJS = ./mpc_solvers/ip_d_box.o
CFLAGS = $(OPT) -std=c99 -fPIC -marm -mfloat-abi=softfp -mfpu=neon -mcpu=cortex-a9 -DTARGET_NEON $(DEBUG)
endif
ifeq ($(TARGET), POWERPC_G2)
AUX_OBJS = ./auxiliary/aux_d_c99.o ./auxiliary/aux_s_c99.o ./auxiliary/block_size_ppc.o 
KERNEL_OBJS_DOUBLE = ./kernel/kernel_dgemm_ppc_g2_lib4.o ./kernel/kernel_dpotrf_c99_lib4.o ./kernel/kernel_dgemv_c99_lib4.o ./kernel/corner_dtrmm_c99_lib4.o ./kernel/corner_dpotrf_c99_lib4.o
KERNEL_OBJS_SINGLE = ./kernel/kernel_sgemm_ppc_g2_lib4.o ./kernel/kernel_spotrf_c99_lib4.o ./kernel/kernel_sgemv_c99_lib4.o ./kernel/corner_strmm_c99_lib4.o ./kernel/corner_spotrf_c99_lib4.o
BLAS_OBJS = ./blas/blas_d_ppc_lib4.o ./blas/blas_s_ppc_lib4.o
LQCP_OBJS = ./lqcp_solvers/dricposv.o ./lqcp_solvers/sricposv.o
LQCP_CODEGEN_OBJS = ./codegen/dricposv_codegen.o ./codegen/sricposv_codegen.o
MPC_OBJS = ./mpc_solvers/ip_d_box.o
CFLAGS = $(OPT) -std=c99 -fPIC -mcpu=603e -DTARGET_POWERPC_G2 $(DEBUG)
endif
ifeq ($(TARGET), C99_4X4)
AUX_OBJS = ./auxiliary/aux_d_c99.o ./auxiliary/aux_s_c99.o ./auxiliary/block_size_c99_4x4.o 
KERNEL_OBJS_DOUBLE = ./kernel/kernel_dgemm_c99_lib4.o ./kernel/kernel_dpotrf_c99_lib4.o ./kernel/kernel_dgemv_c99_lib4.o ./kernel/corner_dtrmm_c99_lib4.o ./kernel/corner_dpotrf_c99_lib4.o 
KERNEL_OBJS_SINGLE = ./kernel/kernel_sgemm_c99_lib4.o ./kernel/kernel_spotrf_c99_lib4.o ./kernel/kernel_sgemv_c99_lib4.o ./kernel/corner_strmm_c99_lib4.o ./kernel/corner_spotrf_c99_lib4.o
BLAS_OBJS = ./blas/blas_d_c99_lib4.o ./blas/blas_s_c99_lib4.o
LQCP_OBJS = ./lqcp_solvers/dricposv.o ./lqcp_solvers/sricposv.o
LQCP_CODEGEN_OBJS = ./codegen/dricposv_codegen.o ./codegen/sricposv_codegen.o
MPC_OBJS = ./mpc_solvers/ip_d_box.o
CFLAGS = $(OPT) -std=c99 -fPIC -DTARGET_C99_4X4 $(DEBUG)
endif
ifeq ($(TARGET), C99_2X2)
AUX_OBJS = ./auxiliary/aux_d_c99.o ./auxiliary/aux_s_c99.o ./auxiliary/block_size_c99_2x2.o 
KERNEL_OBJS_DOUBLE = ./kernel/kernel_dgemm_c99_lib2.o ./kernel/kernel_dpotrf_c99_lib2.o ./kernel/kernel_dgemv_c99_lib2.o ./kernel/corner_dtrmm_c99_lib2.o ./kernel/corner_dpotrf_c99_lib2.o
BLAS_OBJS = ./blas/blas_d_c99_lib2.o
LQCP_OBJS = ./lqcp_solvers/dricposv.o
LQCP_CODEGEN_OBJS = ./codegen/dricposv_codegen.o
MPC_OBJS = ./mpc_solvers/ip_d_box.o
CFLAGS = $(OPT) -std=c99 -fPIC -DTARGET_C99_2X2 $(DEBUG)
endif
ifeq ($(TARGET), ATOM)
AUX_OBJS = ./auxiliary/aux_d_c99.o ./auxiliary/aux_s_c99.o ./auxiliary/block_size_atom.o 
KERNEL_OBJS_DOUBLE = ./kernel/kernel_dgemm_atom_lib2.o ./kernel/kernel_dpotrf_c99_lib2.o ./kernel/kernel_dgemv_c99_lib2.o ./kernel/corner_dtrmm_c99_lib2.o ./kernel/corner_dpotrf_c99_lib2.o
KERNEL_OBJS_SINGLE = ./kernel/kernel_sgemm_atom_lib4.o ./kernel/kernel_spotrf_c99_lib4.o ./kernel/kernel_sgemv_c99_lib4.o ./kernel/corner_strmm_c99_lib4.o ./kernel/corner_spotrf_c99_lib4.o
BLAS_OBJS = ./blas/blas_d_atom_lib2.o ./blas/blas_s_atom_lib4.o
LQCP_OBJS = ./lqcp_solvers/dricposv.o ./lqcp_solvers/sricposv.o
LQCP_CODEGEN_OBJS = ./codegen/dricposv_codegen.o ./codegen/sricposv_codegen.o
MPC_OBJS = ./mpc_solvers/ip_d_box.o
CFLAGS = $(OPT) -std=c99 -fPIC -msse3 -mfpmath=sse -march=atom -DTARGET_ATOM $(DEBUG)
endif

all: clean library test_problem run

code: clean codegenerator test_problem run

library:
	make -C auxiliary obj
	make -C kernel obj
	make -C blas obj
	make -C lqcp_solvers obj
	make -C mpc_solvers obj
	ar rcs HPMPC.a $(AUX_OBJS) $(KERNEL_OBJS_DOUBLE) $(KERNEL_OBJS_SINGLE) $(BLAS_OBJS) $(LQCP_OBJS) $(MPC_OBJS)
	@echo
	@echo " HPMPC.a library build complete."
	@echo

codegenerator:
	make -C auxiliary obj
	make -C kernel obj
#	make -C blas obj
	make -C codegen obj
	make -C mpc_solvers obj
	ar rcs HPMPC.a $(AUX_OBJS) $(KERNEL_OBJS_DOUBLE) $(KERNEL_OBJS_SINGLE) $(LQCP_CODEGEN_OBJS) $(MPC_OBJS)
	@echo
	@echo " HPMPC.a code generator build complete."
	@echo

test_problem:
	cp HPMPC.a ./test_problems/HPMPC.a
#	cp HPMPC.a ./matlab/HPMPC.a
	make -C test_problems obj
	@echo
	@echo " Test problem build complete."
	@echo

run:
	./test_problems/test.out

clean:
	make -C auxiliary clean
	make -C kernel clean
	make -C blas clean
	make -C lqcp_solvers clean
	make -C mpc_solvers clean
	make -C codegen clean
	make -C test_problems clean
#	make -C matlab clean
#	rm -f $(OBJS)
	rm -f test.out
	rm -f *.s
	rm -f *.o
	rm -f HPMPC.a
#	rm -f ./matlab/HPMPC.a

