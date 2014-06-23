###################################################################################################
#                                                                                                 #
# This file is part of HPMPC.                                                                     #
#                                                                                                 #
# HPMPC -- Library for High-Performance implementation of solvers for MPC.                        #
# Copyright (C) 2014 by Technical University of Denmark. All rights reserved.                     #
#                                                                                                 #
# HPMPC is free software; you can redistribute it and/or                                          #
# modify it under the terms of the GNU Lesser General Public                                      #
# License as published by the Free Software Foundation; either                                    #
# version 2.1 of the License, or (at your option) any later version.                              #
#                                                                                                 #
# HPMPC is distributed in the hope that it will be useful,                                        #
# but WITHOUT ANY WARRANTY; without even the implied warranty of                                  #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                                            #
# See the GNU Lesser General Public License for more details.                                     #
#                                                                                                 #
# You should have received a copy of the GNU Lesser General Public                                #
# License along with HPMPC; if not, write to the Free Software                                    #
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA                  #
#                                                                                                 #
# Author: Gianluca Frison, giaf (at) dtu.dk                                                       #
#                                                                                                 #
###################################################################################################


include ./Makefile.rule

ifeq ($(TARGET), X64_AVX)
AUX_OBJS = ./auxiliary/aux_d_c99_lib4.o ./auxiliary/aux_s_c99_lib4.o ./auxiliary/block_size_x64_avx.o 
KERNEL_OBJS_DOUBLE = ./kernel/kernel_dgemm_avx_lib4.o ./kernel/kernel_dpotrf_sse3_lib4.o ./kernel/kernel_dgemv_avx_lib4.o ./kernel/corner_dtrmm_avx_lib4.o ./kernel/corner_dpotrf_sse2_lib4.o ./kernel/kernel_dsymv_avx_lib4.o 
KERNEL_OBJS_SINGLE = ./kernel/kernel_sgemm_avx_lib4.o ./kernel/kernel_spotrf_sse_lib4.o ./kernel/kernel_sgemv_sse_lib4.o ./kernel/corner_strmm_sse_lib4.o ./kernel/corner_spotrf_c99_lib4.o ./kernel/kernel_ssymv_sse_lib4.o
BLAS_OBJS = ./blas/blas_d_lib4.o ./blas/blas_s_lib4.o
endif
ifeq ($(TARGET), X64_SSE3)
AUX_OBJS = ./auxiliary/aux_d_c99_lib4.o ./auxiliary/aux_s_c99_lib4.o ./auxiliary/block_size_x64_sse.o 
KERNEL_OBJS_DOUBLE = ./kernel/kernel_dgemm_sse3_lib4.o ./kernel/kernel_dpotrf_sse3_lib4.o ./kernel/kernel_dgemv_sse3_lib4.o ./kernel/corner_dtrmm_sse3_lib4.o ./kernel/corner_dpotrf_sse2_lib4.o ./kernel/kernel_dsymv_c99_lib4.o 
KERNEL_OBJS_SINGLE = ./kernel/kernel_sgemm_sse_lib4.o ./kernel/kernel_spotrf_sse_lib4.o ./kernel/kernel_sgemv_sse_lib4.o ./kernel/corner_strmm_sse_lib4.o ./kernel/corner_spotrf_c99_lib4.o ./kernel/kernel_ssymv_sse_lib4.o
BLAS_OBJS = ./blas/blas_d_lib4.o ./blas/blas_s_lib4.o
endif
ifeq ($(TARGET), AMD_SSE3)
AUX_OBJS = ./auxiliary/aux_d_c99_lib4.o ./auxiliary/aux_s_c99_lib4.o ./auxiliary/block_size_x64_sse.o 
KERNEL_OBJS_DOUBLE = ./kernel/kernel_dgemm_amd_sse3_lib4.o ./kernel/kernel_dpotrf_sse3_lib4.o ./kernel/kernel_dgemv_sse3_lib4.o ./kernel/corner_dtrmm_sse3_lib4.o ./kernel/corner_dpotrf_sse2_lib4.o ./kernel/kernel_dsymv_c99_lib4.o 
KERNEL_OBJS_SINGLE = ./kernel/kernel_sgemm_sse_lib4.o ./kernel/kernel_spotrf_sse_lib4.o ./kernel/kernel_sgemv_sse_lib4.o ./kernel/corner_strmm_sse_lib4.o ./kernel/corner_spotrf_c99_lib4.o ./kernel/kernel_ssymv_sse_lib4.o
BLAS_OBJS = ./blas/blas_d_lib4.o ./blas/blas_s_lib4.o
endif
ifeq ($(TARGET), NEON)
AUX_OBJS = ./auxiliary/aux_d_c99_lib4.o ./auxiliary/aux_s_c99_lib4.o ./auxiliary/block_size_neon.o 
KERNEL_OBJS_DOUBLE = ./kernel/kernel_dgemm_neon_lib4.o ./kernel/kernel_dpotrf_c99_lib4.o ./kernel/kernel_dgemv_c99_lib4.o ./kernel/corner_dtrmm_c99_lib4.o ./kernel/corner_dpotrf_c99_lib4.o ./kernel/kernel_dsymv_c99_lib4.o 
KERNEL_OBJS_SINGLE = ./kernel/kernel_sgemm_neon_lib4.o ./kernel/kernel_spotrf_c99_lib4.o ./kernel/kernel_sgemv_c99_lib4.o ./kernel/corner_strmm_c99_lib4.o ./kernel/corner_spotrf_c99_lib4.o ./kernel/kernel_ssymv_c99_lib4.o
BLAS_OBJS = ./blas/blas_d_lib4.o ./blas/blas_s_lib4.o
endif
ifeq ($(TARGET), POWERPC_G2)
AUX_OBJS = ./auxiliary/aux_d_c99_lib4.o ./auxiliary/aux_s_c99_lib4.o ./auxiliary/block_size_ppc.o 
KERNEL_OBJS_DOUBLE = ./kernel/kernel_dgemm_ppc_g2_lib4.o ./kernel/kernel_dpotrf_c99_lib4.o ./kernel/kernel_dgemv_c99_lib4.o ./kernel/corner_dtrmm_c99_lib4.o ./kernel/corner_dpotrf_c99_lib4.o ./kernel/kernel_dsymv_c99_lib4.o 
KERNEL_OBJS_SINGLE = ./kernel/kernel_sgemm_ppc_g2_lib4.o ./kernel/kernel_spotrf_c99_lib4.o ./kernel/kernel_sgemv_c99_lib4.o ./kernel/corner_strmm_c99_lib4.o ./kernel/corner_spotrf_c99_lib4.o ./kernel/kernel_ssymv_c99_lib4.o
BLAS_OBJS = ./blas/blas_d_lib4.o ./blas/blas_s_lib4.o
endif
ifeq ($(TARGET), C99_4X4)
AUX_OBJS = ./auxiliary/aux_d_c99_lib4.o ./auxiliary/aux_s_c99_lib4.o ./auxiliary/block_size_c99_4x4.o 
KERNEL_OBJS_DOUBLE = ./kernel/kernel_dgemm_c99_lib4.o ./kernel/kernel_dpotrf_c99_lib4.o ./kernel/kernel_dgemv_c99_lib4.o ./kernel/corner_dtrmm_c99_lib4.o ./kernel/corner_dpotrf_c99_lib4.o ./kernel/kernel_dsymv_c99_lib4.o 
KERNEL_OBJS_SINGLE = ./kernel/kernel_sgemm_c99_lib4.o ./kernel/kernel_spotrf_c99_lib4.o ./kernel/kernel_sgemv_c99_lib4.o ./kernel/corner_strmm_c99_lib4.o ./kernel/corner_spotrf_c99_lib4.o ./kernel/kernel_ssymv_c99_lib4.o
BLAS_OBJS = ./blas/blas_d_lib4.o ./blas/blas_s_lib4.o
endif
ifeq ($(TARGET), C99_2X2)
AUX_OBJS = ./auxiliary/aux_d_c99_lib2.o ./auxiliary/aux_s_c99_lib2.o ./auxiliary/block_size_c99_2x2.o 
KERNEL_OBJS_DOUBLE = ./kernel/kernel_dgemm_c99_lib2.o ./kernel/kernel_dpotrf_c99_lib2.o ./kernel/kernel_dgemv_c99_lib2.o ./kernel/corner_dtrmm_c99_lib2.o ./kernel/corner_dpotrf_c99_lib2.o ./kernel/kernel_dsymv_c99_lib2.o
KERNEL_OBJS_SINGLE = ./kernel/kernel_sgemm_c99_lib2.o ./kernel/kernel_spotrf_c99_lib2.o ./kernel/kernel_sgemv_c99_lib2.o ./kernel/corner_strmm_c99_lib2.o ./kernel/corner_spotrf_c99_lib2.o ./kernel/kernel_ssymv_c99_lib2.o
BLAS_OBJS = ./blas/blas_d_lib2.o ./blas/blas_s_lib2.o
endif
ifeq ($(TARGET), X86_ATOM)
AUX_OBJS = ./auxiliary/aux_d_c99_lib2.o ./auxiliary/aux_s_c99_lib2.o ./auxiliary/block_size_x86_atom.o 
KERNEL_OBJS_DOUBLE = ./kernel/kernel_dgemm_x86_atom_lib2.o ./kernel/kernel_dpotrf_c99_lib2.o ./kernel/kernel_dgemv_c99_lib2.o ./kernel/corner_dtrmm_c99_lib2.o ./kernel/corner_dpotrf_c99_lib2.o ./kernel/kernel_dsymv_c99_lib2.o
KERNEL_OBJS_SINGLE = ./kernel/kernel_sgemm_x86_atom_lib4.o ./kernel/kernel_spotrf_c99_lib4.o ./kernel/kernel_sgemv_x86_atom_lib4.o ./kernel/corner_strmm_c99_lib4.o ./kernel/corner_spotrf_c99_lib4.o ./kernel/kernel_ssymv_c99_lib4.o
BLAS_OBJS = ./blas/blas_d_lib2.o ./blas/blas_s_lib4.o
endif
LQCP_OBJS = ./lqcp_solvers/dricposv.o ./lqcp_solvers/sricposv.o ./lqcp_solvers/dres.o ./lqcp_solvers/sres.o
LQCP_CODEGEN_OBJS = ./codegen/dricposv_codegen.o ./codegen/sricposv_codegen.o  ./codegen/dres_codegen.o ./codegen/sres_codegen.o 
MPC_OBJS = ./mpc_solvers/d_ip_box.o ./mpc_solvers/d_ip2_box.o ./mpc_solvers/dres_ip_box.o ./mpc_solvers/s_ip_box.o ./mpc_solvers/s_ip2_box.o ./mpc_solvers/sres_ip_box.o

all: notice clean library test_problem run

codegen: notice clean codegenerator test_problem run

notice:
	@echo
	@echo
	@echo 
	@echo " HPMPC -- Library for High-Performance implementation of solvers for MPC."
	@echo " Copyright (C) 2014 by Technical University of Denmark. All rights reserved."
	@echo
	@echo " HPMPC is distributed in the hope that it will be useful,"
	@echo " but WITHOUT ANY WARRANTY; without even the implied warranty of"
	@echo " MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE."
	@echo " See the GNU Lesser General Public License for more details."
	@echo
	@echo

library:
	@echo
	@echo " Building HPMPC library..."
	@echo
	make -C auxiliary obj
	make -C kernel obj
	make -C blas obj
	make -C lqcp_solvers obj
	make -C mpc_solvers obj
	ar rcs libhpmpc.a $(AUX_OBJS) $(KERNEL_OBJS_DOUBLE) $(KERNEL_OBJS_SINGLE) $(BLAS_OBJS) $(LQCP_OBJS) $(MPC_OBJS)
	@echo
	@echo " HPMPC library build completed."
	@echo

codegenerator:
	@echo
	@echo " Building HPMPC code generator..."
	@echo
	make -C auxiliary obj
	make -C kernel obj
	make -C blas obj
	touch ./codegen/dricposv_codegen.c 
	touch ./codegen/sricposv_codegen.c
	touch ./codegen/dres_codegen.c 
	touch ./codegen/sres_codegen.c 
	make -C codegen obj
	make -C mpc_solvers obj
	ar rcs libhpmpc.a $(AUX_OBJS) $(KERNEL_OBJS_DOUBLE) $(KERNEL_OBJS_SINGLE) $(LQCP_CODEGEN_OBJS) $(BLAS_OBJS) $(MPC_OBJS)
	@echo
	@echo " HPMPC code generator build completed."
	@echo

test_problem:
	@echo
	@echo " Building test problem..."
	@echo
	cp libhpmpc.a ./test_problems/libhpmpc.a
#	cp HPMPC.a ./matlab/HPMPC.a
	make -C test_problems obj
	@echo
	@echo " Test problem build completed."
	@echo

run:
	@echo
	@echo " Running test problem..."
	@echo
	./test_problems/test.out

#install: library
install:
	@echo
	@echo " Installing HPMPC library..."
	@echo
	cp -f libhpmpc.a /usr/lib/libhpmpc.a
	mkdir -p /usr/include/hpmpc
	cp -rf ./include/* /usr/include/hpmpc
	@echo
	@echo " HPMPC library installation completed"
	@echo
	
uninstall:
	@echo
	@echo " Uninstalling HPMPC library..."
	@echo
	rm /lib/libhpmpc.a
	rm -r /include/hpmpc
	@echo
	@echo " HPMPC library uninstallation completed"
	@echo
	
clean:
	@echo
	@echo " Cleaning HPMPC library..."
	@echo
	make -C auxiliary clean
	make -C kernel clean
	make -C blas clean
	make -C lqcp_solvers clean
	make -C mpc_solvers clean
	make -C codegen clean
	make -C test_problems clean
	make -C interfaces/octave clean
	rm -f test.out
	rm -f *.s
	rm -f *.o
	rm -f libhpmpc.a
#	rm -f ./matlab/HPMPC.a
	@echo
	@echo " HPMPC library clean completed."
	@echo

