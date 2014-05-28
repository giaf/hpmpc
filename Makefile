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
AUX_OBJS = ./auxiliary/block_size_x64_avx.o ./auxiliary/aux_d_c99_lib4.o #./auxiliary/aux_s_c99_lib4.o 
KERNEL_OBJS_DOUBLE = ./kernel/kernel_dgemm_avx_lib4.o ./kernel/kernel_dtrmm_avx_lib4.o ./kernel/kernel_dtrsm_avx_lib4.o ./kernel/kernel_dpotrf_avx_lib4.o ./kernel/kernel_dgemv_avx_lib4.o ./kernel/kernel_dtrmv_avx_lib4.o ./kernel/kernel_dtrsv_avx_lib4.o ./kernel/corner_dtrmm_avx_lib4.o ./kernel/kernel_dsymv_avx_lib4.o 
KERNEL_OBJS_SINGLE = #./kernel/kernel_sgemm_avx_lib8.o
BLAS_OBJS = ./blas/blas_d_lib4.o #./blas/blas_s_lib8.o
CFLAGS = $(OPT) -std=c99 -mavx -DTARGET_X64_AVX $(DEBUG)
endif
LQCP_OBJS = ./lqcp_solvers/dricposv.o ./lqcp_solvers/dres.o #./lqcp_solvers/sricposv.o ./lqcp_solvers/sres.o
LQCP_CODEGEN_OBJS = ./codegen/dricposv_codegen.o ./codegen/dres_codegen.o #./codegen/sricposv_codegen.o  ./codegen/sres_codegen.o 
MPC_OBJS = #./mpc_solvers/d_ip_box.o ./mpc_solvers/d_ip2_box.o ./mpc_solvers/dres_ip_box.o ./mpc_solvers/s_ip_box.o ./mpc_solvers/s_ip2_box.o ./mpc_solvers/sres_ip_box.o

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

