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
AUX_OBJS = ./auxiliary/block_size_x64_avx.o ./auxiliary/aux_d_c99_lib4.o ./auxiliary/aux_s_c99.o 
KERNEL_OBJS_DOUBLE = ./kernel/avx/kernel_dgemm_avx_lib4.o ./kernel/avx/kernel_dtrmm_avx_lib4.o ./kernel/avx/kernel_dtrsm_avx_lib4.o ./kernel/avx/kernel_dpotrf_avx_lib4.o ./kernel/avx/kernel_dgemv_avx_lib4.o ./kernel/avx/kernel_dtrmv_avx_lib4.o ./kernel/avx/kernel_dtrsv_avx_lib4.o ./kernel/avx/kernel_dsymv_avx_lib4.o ./kernel/avx/kernel_dtran_avx_lib4.o 
KERNEL_OBJS_SINGLE = ./kernel/avx/kernel_sgemm_avx_lib8.o
BLAS_OBJS = ./blas/blas_d_lib4.o ./blas/blas_s_lib8.o
MPC_OBJS = ./mpc_solvers/d_ip_box.o ./mpc_solvers/dres_ip_box.o ./mpc_solvers/d_aux_ip_avx.o ./mpc_solvers/d_ip2_box.o #./mpc_solvers/s_ip_box.o ./mpc_solvers/s_ip2_box.o ./mpc_solvers/sres_ip_box.o
CFLAGS = $(OPT) -std=c99 -mavx -DTARGET_X64_AVX -fPIC $(DEBUG)
endif
ifeq ($(TARGET), C99_4X4)
AUX_OBJS = ./auxiliary/block_size_c99_4x4.o ./auxiliary/aux_d_c99_lib4.o ./auxiliary/aux_s_c99.o 
KERNEL_OBJS_DOUBLE = ./kernel/c99/kernel_dgemm_c99_lib4.o ./kernel/c99/kernel_dtrmm_c99_lib4.o ./kernel/c99/kernel_dtrsm_c99_lib4.o ./kernel/c99/kernel_dpotrf_c99_lib4.o ./kernel/c99/kernel_dgemv_c99_lib4.o ./kernel/c99/kernel_dtrmv_c99_lib4.o ./kernel/c99/kernel_dtrsv_c99_lib4.o ./kernel/c99/kernel_dsymv_c99_lib4.o ./kernel/c99/kernel_dtran_c99_lib4.o 
KERNEL_OBJS_SINGLE = ./kernel/c99/kernel_sgemm_c99_lib4.o
BLAS_OBJS = ./blas/blas_d_lib4.o ./blas/blas_s_lib4.o
MPC_OBJS = #./mpc_solvers/d_ip_box.o ./mpc_solvers/dres_ip_box.o ./mpc_solvers/d_aux_ip_avx.o ./mpc_solvers/d_ip2_box.o #./mpc_solvers/s_ip_box.o ./mpc_solvers/s_ip2_box.o ./mpc_solvers/sres_ip_box.o
CFLAGS = $(OPT) -std=c99 -DTARGET_C99_4X4 -fPIC $(DEBUG)
endif
ifeq ($(TARGET), CORTEX_A15)
AUX_OBJS = ./auxiliary/block_size_cortex_a15.o ./auxiliary/aux_d_c99_lib4.o ./auxiliary/aux_s_c99.o 
KERNEL_OBJS_DOUBLE = ./kernel/neon/kernel_dgemm_vfpv3_lib4.o ./kernel/neon/kernel_dtrmm_c99_lib4.o ./kernel/neon/kernel_dtrsm_c99_lib4.o ./kernel/neon/kernel_dpotrf_c99_lib4.o ./kernel/neon/kernel_dgemv_c99_lib4.o ./kernel/neon/kernel_dtrmv_c99_lib4.o ./kernel/neon/kernel_dtrsv_c99_lib4.o ./kernel/neon/kernel_dsymv_c99_lib4.o ./kernel/neon/kernel_dtran_c99_lib4.o 
KERNEL_OBJS_SINGLE = ./kernel/neon/kernel_sgemm_neon_lib4.o
BLAS_OBJS = ./blas/blas_d_lib4.o ./blas/blas_s_lib4.o
MPC_OBJS = #./mpc_solvers/d_ip_box.o ./mpc_solvers/dres_ip_box.o ./mpc_solvers/d_aux_ip_avx.o ./mpc_solvers/d_ip2_box.o #./mpc_solvers/s_ip_box.o ./mpc_solvers/s_ip2_box.o ./mpc_solvers/sres_ip_box.o
CFLAGS = $(OPT) -std=c99 -DTARGET_CORTEX_A15 -marm -mfloat-abi=hard -mfpu=neon -mcpu=cortex-a15 -fPIC $(DEBUG)
endif
LQCP_OBJS = ./lqcp_solvers/dricposv.o ./lqcp_solvers/dres.o #./lqcp_solvers/sricposv.o ./lqcp_solvers/sres.o
LQCP_CODEGEN_OBJS = ./codegen/dricposv_codegen.o ./codegen/dres_codegen.o #./codegen/sricposv_codegen.o  ./codegen/sres_codegen.o 

all: clean library test_problem run

codegen: clean codegenerator test_problem run

library:
	make -C auxiliary obj
	make -C kernel obj
	make -C blas obj
	make -C lqcp_solvers obj
	make -C mpc_solvers obj
	ar rcs libhpmpc.a $(AUX_OBJS) $(KERNEL_OBJS_DOUBLE) $(KERNEL_OBJS_SINGLE) $(BLAS_OBJS) $(LQCP_OBJS) $(MPC_OBJS)
	@echo
	@echo " libhpmpc.a library build complete."
	@echo

codegenerator:
	make -C auxiliary obj
	make -C kernel obj
	make -C blas obj
	touch ./codegen/dricposv_codegen.c 
#	touch ./codegen/sricposv_codegen.c
	touch ./codegen/dres_codegen.c 
#	touch ./codegen/sres_codegen.c 
	make -C codegen obj
	make -C mpc_solvers obj
	ar rcs libhpmpc.a $(AUX_OBJS) $(KERNEL_OBJS_DOUBLE) $(KERNEL_OBJS_SINGLE) $(BLAS_OBJS) $(LQCP_CODEGEN_OBJS) $(MPC_OBJS)
	@echo
	@echo " libhpmpc.a code generator build complete."
	@echo

test_problem:
	cp libhpmpc.a ./test_problems/libhpmpc.a
#	cp HPMPC.a ./matlab/HPMPC.a
	make -C test_problems obj
	@echo
	@echo " Test problem build complete."
	@echo

run:
	./test_problems/test.out

#install: library
install:
	cp -f libhpmpc.a /usr/local/lib/libhpmpc.a
	cp -rf ./include /usr/local/include/hpmpc
	
uninstall:
	rm /lib/libhpmpc.a
	rm -r /include/hpmpc
	
clean:
	make -C auxiliary clean
	make -C kernel clean
	make -C blas clean
	make -C lqcp_solvers clean
	make -C mpc_solvers clean
	make -C codegen clean
	make -C test_problems clean
	make -C interfaces/octave clean
#	rm -f $(OBJS)
	rm -f test.out
	rm -f *.s
	rm -f *.o
	rm -f libhpmpc.a
#	rm -f ./matlab/HPMPC.a

