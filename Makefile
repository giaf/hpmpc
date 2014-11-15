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

ifeq ($(TARGET), X64_AVX2)
AUX_OBJS = ./auxiliary/block_size_x64_avx.o ./auxiliary/aux_d_c99_lib4.o ./auxiliary/aux_s_c99_lib8.o 
KERNEL_OBJS_DOUBLE = ./kernel/avx2/kernel_dgemm_avx2_lib4.o ./kernel/avx2/kernel_dtrmm_avx_lib4.o ./kernel/avx2/kernel_dtrsm_avx_lib4.o ./kernel/avx2/kernel_dgemm_dtrsm_avx_lib4.o ./kernel/avx2/kernel_dsyrk_avx_lib4.o ./kernel/avx2/kernel_dpotrf_avx_lib4.o ./kernel/avx2/kernel_dsyrk_dpotrf_avx_lib4.o ./kernel/avx2/kernel_dgemv_avx_lib4.o ./kernel/avx2/kernel_dtrmv_avx_lib4.o ./kernel/avx2/kernel_dtrsv_avx_lib4.o ./kernel/avx2/kernel_dsymv_avx_lib4.o ./kernel/avx2/kernel_dtran_avx_lib4.o ./kernel/avx2/kernel_dttmm_c99_lib4.o 
KERNEL_OBJS_SINGLE = ./kernel/avx2/kernel_sgemm_avx2_lib8.o ./kernel/avx2/kernel_strmm_avx2_lib8.o ./kernel/avx2/kernel_sgemm_strsm_avx2_lib8.o ./kernel/avx2/kernel_ssyrk_spotrf_avx2_lib8.o ./kernel/avx2/kernel_sgemv_avx_lib8.o ./kernel/avx2/kernel_strmv_avx_lib8.o ./kernel/avx2/kernel_strsv_avx_lib8.o ./kernel/avx2/kernel_ssymv_avx_lib8.o ./kernel/avx2/kernel_stran_avx2_lib8.o
BLAS_OBJS = ./blas/blas_d_lib4.o ./blas/blas_s_lib8.o
MPC_OBJS = ./mpc_solvers/d_ip_box.o ./mpc_solvers/d_res_ip_box.o ./mpc_solvers/avx/d_aux_ip_avx_lib4.o ./mpc_solvers/d_ip2_box.o ./mpc_solvers/d_admm_box.o ./mpc_solvers/d_admm_soft.o ./mpc_solvers/s_ip_box.o ./mpc_solvers/s_res_ip_box.o ./mpc_solvers/avx/s_aux_ip_avx_lib8.o ./mpc_solvers/s_ip2_box.o ./mpc_solvers/s_admm_box.o ./mpc_solvers/s_admm_soft.o
#CFLAGS = $(OPT) -std=c99 -mavx -DTARGET_X64_AVX -fPIC $(DEBUG)
endif
ifeq ($(TARGET), X64_AVX)
AUX_OBJS = ./auxiliary/block_size_x64_avx.o ./auxiliary/aux_d_c99_lib4.o ./auxiliary/aux_s_c99_lib8.o 
KERNEL_OBJS_DOUBLE = ./kernel/avx/kernel_dgemm_avx_lib4.o ./kernel/avx/kernel_dtrmm_avx_lib4.o ./kernel/avx/kernel_dtrsm_avx_lib4.o ./kernel/avx/kernel_dgemm_dtrsm_avx_lib4.o ./kernel/avx/kernel_dsyrk_avx_lib4.o ./kernel/avx/kernel_dpotrf_avx_lib4.o ./kernel/avx/kernel_dsyrk_dpotrf_avx_lib4.o ./kernel/avx/kernel_dgemv_avx_lib4.o ./kernel/avx/kernel_dtrmv_avx_lib4.o ./kernel/avx/kernel_dtrsv_avx_lib4.o ./kernel/avx/kernel_dsymv_avx_lib4.o ./kernel/avx/kernel_dtran_avx_lib4.o ./kernel/avx/kernel_dttmm_c99_lib4.o 
KERNEL_OBJS_SINGLE = ./kernel/avx/kernel_sgemm_avx_lib8.o ./kernel/avx/kernel_strmm_avx_lib8.o ./kernel/avx/kernel_sgemm_strsm_avx_lib8.o ./kernel/avx/kernel_ssyrk_spotrf_avx_lib8.o ./kernel/avx/kernel_sgemv_avx_lib8.o ./kernel/avx/kernel_strmv_avx_lib8.o ./kernel/avx/kernel_strsv_avx_lib8.o ./kernel/avx/kernel_ssymv_avx_lib8.o ./kernel/avx/kernel_stran_avx_lib8.o  
BLAS_OBJS = ./blas/blas_d_lib4.o ./blas/blas_s_lib8.o
MPC_OBJS = ./mpc_solvers/d_ip_box.o ./mpc_solvers/d_res_ip_box.o ./mpc_solvers/avx/d_aux_ip_avx_lib4.o ./mpc_solvers/d_ip2_box.o ./mpc_solvers/d_admm_box.o ./mpc_solvers/d_admm_soft.o ./mpc_solvers/s_ip_box.o ./mpc_solvers/s_res_ip_box.o ./mpc_solvers/avx/s_aux_ip_avx_lib8.o ./mpc_solvers/s_ip2_box.o ./mpc_solvers/s_admm_box.o ./mpc_solvers/s_admm_soft.o
#CFLAGS = $(OPT) -std=c99 -mavx -DTARGET_X64_AVX -fPIC $(DEBUG)
endif
ifeq ($(TARGET), X64_SSE3)
AUX_OBJS = ./auxiliary/block_size_x64_sse.o ./auxiliary/aux_d_c99_lib4.o ./auxiliary/aux_s_c99_lib4.o 
KERNEL_OBJS_DOUBLE = ./kernel/sse3/kernel_dgemm_sse3_lib4.o ./kernel/sse3/kernel_dtrmm_sse3_lib4.o ./kernel/sse3/kernel_dtrsm_sse3_lib4.o ./kernel/sse3/kernel_dgemm_dtrsm_sse3_lib4.o ./kernel/sse3/kernel_dsyrk_sse3_lib4.o ./kernel/sse3/kernel_dpotrf_sse3_lib4.o ./kernel/sse3/kernel_dsyrk_dpotrf_sse3_lib4.o ./kernel/sse3/kernel_dgemv_c99_lib4.o ./kernel/sse3/kernel_dtrmv_c99_lib4.o ./kernel/sse3/kernel_dtrsv_c99_lib4.o ./kernel/sse3/kernel_dsymv_c99_lib4.o ./kernel/sse3/kernel_dtran_c99_lib4.o ./kernel/sse3/kernel_dttmm_c99_lib4.o 
KERNEL_OBJS_SINGLE = ./kernel/sse3/kernel_sgemm_sse_lib4.o ./kernel/sse3/kernel_strmm_sse_lib4.o ./kernel/sse3/kernel_sgemm_strsm_sse_lib4.o ./kernel/sse3/kernel_ssyrk_spotrf_sse_lib4.o ./kernel/sse3/kernel_sgemv_c99_lib4.o ./kernel/sse3/kernel_strmv_c99_lib4.o ./kernel/sse3/kernel_strsv_c99_lib4.o ./kernel/sse3/kernel_ssymv_c99_lib4.o ./kernel/sse3/kernel_stran_c99_lib4.o 
BLAS_OBJS = ./blas/blas_d_lib4.o ./blas/blas_s_lib4.o
MPC_OBJS = ./mpc_solvers/d_ip_box.o ./mpc_solvers/d_res_ip_box.o ./mpc_solvers/c99/d_aux_ip_c99_lib4.o ./mpc_solvers/d_ip2_box.o ./mpc_solvers/d_admm_box.o ./mpc_solvers/d_admm_soft.o ./mpc_solvers/s_ip_box.o ./mpc_solvers/s_res_ip_box.o ./mpc_solvers/c99/s_aux_ip_c99_lib4.o ./mpc_solvers/s_ip2_box.o ./mpc_solvers/s_admm_box.o ./mpc_solvers/s_admm_soft.o
#CFLAGS = $(OPT) -std=c99 -DTARGET_C99_4X4 -fPIC $(DEBUG)
endif
ifeq ($(TARGET), C99_4X4)
AUX_OBJS = ./auxiliary/block_size_c99_4x4.o ./auxiliary/aux_d_c99_lib4.o ./auxiliary/aux_s_c99_lib4.o 
KERNEL_OBJS_DOUBLE = ./kernel/c99/kernel_dgemm_c99_lib4.o ./kernel/c99/kernel_dtrmm_c99_lib4.o ./kernel/c99/kernel_dtrsm_c99_lib4.o ./kernel/c99/kernel_dgemm_dtrsm_c99_lib4.o ./kernel/c99/kernel_dsyrk_c99_lib4.o ./kernel/c99/kernel_dpotrf_c99_lib4.o ./kernel/c99/kernel_dsyrk_dpotrf_c99_lib4.o ./kernel/c99/kernel_dgemv_c99_lib4.o ./kernel/c99/kernel_dtrmv_c99_lib4.o ./kernel/c99/kernel_dtrsv_c99_lib4.o ./kernel/c99/kernel_dsymv_c99_lib4.o ./kernel/c99/kernel_dtran_c99_lib4.o ./kernel/c99/kernel_dttmm_c99_lib4.o
KERNEL_OBJS_SINGLE = ./kernel/c99/kernel_sgemm_c99_lib4.o ./kernel/c99/kernel_strmm_c99_lib4.o ./kernel/c99/kernel_sgemm_strsm_c99_lib4.o ./kernel/c99/kernel_ssyrk_spotrf_c99_lib4.o ./kernel/c99/kernel_sgemv_c99_lib4.o ./kernel/c99/kernel_strmv_c99_lib4.o ./kernel/c99/kernel_strsv_c99_lib4.o ./kernel/c99/kernel_ssymv_c99_lib4.o ./kernel/c99/kernel_stran_c99_lib4.o 
BLAS_OBJS = ./blas/blas_d_lib4.o ./blas/blas_s_lib4.o
MPC_OBJS = ./mpc_solvers/d_ip_box.o ./mpc_solvers/d_res_ip_box.o ./mpc_solvers/c99/d_aux_ip_c99_lib4.o ./mpc_solvers/d_ip2_box.o ./mpc_solvers/d_admm_box.o ./mpc_solvers/d_admm_soft.o ./mpc_solvers/s_ip_box.o ./mpc_solvers/s_res_ip_box.o ./mpc_solvers/c99/s_aux_ip_c99_lib4.o ./mpc_solvers/s_ip2_box.o ./mpc_solvers/s_admm_box.o ./mpc_solvers/s_admm_soft.o
#CFLAGS = $(OPT) -std=c99 -DTARGET_C99_4X4 -fPIC $(DEBUG)
endif
ifeq ($(TARGET), CORTEX_A15)
AUX_OBJS = ./auxiliary/block_size_cortex_a15.o ./auxiliary/aux_d_c99_lib4.o ./auxiliary/aux_s_c99_lib4.o 
KERNEL_OBJS_DOUBLE = ./kernel/neon/kernel_dgemm_vfpv3_lib4.o ./kernel/neon/kernel_dtrmm_vfpv3_lib4.o ./kernel/neon/kernel_dtrsm_vfpv3_lib4.o ./kernel/neon/kernel_dgemm_dtrsm_vfpv3_lib4.o ./kernel/neon/kernel_dsyrk_vfpv3_lib4.o ./kernel/neon/kernel_dpotrf_vfpv3_lib4.o ./kernel/neon/kernel_dsyrk_dpotrf_vfpv3_lib4.o ./kernel/neon/kernel_dgemv_c99_lib4.o ./kernel/neon/kernel_dtrmv_c99_lib4.o ./kernel/neon/kernel_dtrsv_c99_lib4.o ./kernel/neon/kernel_dsymv_c99_lib4.o ./kernel/neon/kernel_dtran_c99_lib4.o ./kernel/neon/kernel_dttmm_c99_lib4.o 
KERNEL_OBJS_SINGLE = ./kernel/neon/kernel_sgemm_neon_lib4.o ./kernel/neon/kernel_strmm_neon_lib4.o ./kernel/neon/kernel_sgemm_strsm_neon_lib4.o ./kernel/neon/kernel_ssyrk_spotrf_neon_lib4.o ./kernel/neon/kernel_sgemv_neon_lib4.o ./kernel/neon/kernel_strmv_neon_lib4.o ./kernel/neon/kernel_strsv_neon_lib4.o ./kernel/neon/kernel_ssymv_c99_lib4.o ./kernel/neon/kernel_stran_neon_lib4.o 
BLAS_OBJS = ./blas/blas_d_lib4.o ./blas/blas_s_lib4.o
MPC_OBJS = ./mpc_solvers/d_ip_box.o ./mpc_solvers/d_res_ip_box.o ./mpc_solvers/c99/d_aux_ip_c99_lib4.o ./mpc_solvers/d_ip2_box.o ./mpc_solvers/d_admm_box.o ./mpc_solvers/d_admm_soft.o ./mpc_solvers/s_ip_box.o ./mpc_solvers/s_res_ip_box.o ./mpc_solvers/c99/s_aux_ip_c99_lib4.o ./mpc_solvers/s_ip2_box.o ./mpc_solvers/s_admm_box.o ./mpc_solvers/s_admm_soft.o
endif
ifeq ($(TARGET), CORTEX_A9)
AUX_OBJS = ./auxiliary/block_size_cortex_a15.o ./auxiliary/aux_d_c99_lib4.o ./auxiliary/aux_s_c99_lib4.o 
KERNEL_OBJS_DOUBLE = ./kernel/neon/kernel_dgemm_vfpv3_lib4.o ./kernel/neon/kernel_dtrmm_vfpv3_lib4.o ./kernel/neon/kernel_dtrsm_vfpv3_lib4.o ./kernel/neon/kernel_dgemm_dtrsm_vfpv3_lib4.o ./kernel/neon/kernel_dsyrk_vfpv3_lib4.o ./kernel/neon/kernel_dpotrf_vfpv3_lib4.o ./kernel/neon/kernel_dsyrk_dpotrf_vfpv3_lib4.o ./kernel/neon/kernel_dgemv_c99_lib4.o ./kernel/neon/kernel_dtrmv_c99_lib4.o ./kernel/neon/kernel_dtrsv_c99_lib4.o ./kernel/neon/kernel_dsymv_c99_lib4.o ./kernel/neon/kernel_dtran_c99_lib4.o ./kernel/neon/kernel_dttmm_c99_lib4.o 
KERNEL_OBJS_SINGLE = ./kernel/neon/kernel_sgemm_neon_lib4.o  ./kernel/neon/kernel_strmm_neon_lib4.o ./kernel/neon/kernel_sgemm_strsm_neon_lib4.o ./kernel/neon/kernel_ssyrk_spotrf_neon_lib4.o ./kernel/neon/kernel_sgemv_neon_lib4.o ./kernel/neon/kernel_strmv_neon_lib4.o ./kernel/neon/kernel_strsv_neon_lib4.o ./kernel/neon/kernel_ssymv_c99_lib4.o ./kernel/neon/kernel_stran_neon_lib4.o 
BLAS_OBJS = ./blas/blas_d_lib4.o ./blas/blas_s_lib4.o
MPC_OBJS = ./mpc_solvers/d_ip_box.o ./mpc_solvers/d_res_ip_box.o ./mpc_solvers/c99/d_aux_ip_c99_lib4.o ./mpc_solvers/d_ip2_box.o ./mpc_solvers/d_admm_box.o ./mpc_solvers/d_admm_soft.o ./mpc_solvers/s_ip_box.o ./mpc_solvers/s_res_ip_box.o ./mpc_solvers/c99/s_aux_ip_c99_lib4.o ./mpc_solvers/s_ip2_box.o ./mpc_solvers/s_admm_box.o ./mpc_solvers/s_admm_soft.o
endif
ifeq ($(TARGET), CORTEX_A7)
AUX_OBJS = ./auxiliary/block_size_cortex_a15.o ./auxiliary/aux_d_c99_lib4.o ./auxiliary/aux_s_c99_lib4.o 
KERNEL_OBJS_DOUBLE = ./kernel/neon/kernel_dgemm_vfpv3_lib4.o ./kernel/neon/kernel_dtrmm_vfpv3_lib4.o ./kernel/neon/kernel_dtrsm_vfpv3_lib4.o ./kernel/neon/kernel_dgemm_dtrsm_vfpv3_lib4.o ./kernel/neon/kernel_dsyrk_vfpv3_lib4.o ./kernel/neon/kernel_dpotrf_vfpv3_lib4.o ./kernel/neon/kernel_dsyrk_dpotrf_vfpv3_lib4.o ./kernel/neon/kernel_dgemv_c99_lib4.o ./kernel/neon/kernel_dtrmv_c99_lib4.o ./kernel/neon/kernel_dtrsv_c99_lib4.o ./kernel/neon/kernel_dsymv_c99_lib4.o ./kernel/neon/kernel_dtran_c99_lib4.o ./kernel/neon/kernel_dttmm_c99_lib4.o 
KERNEL_OBJS_SINGLE = ./kernel/neon/kernel_sgemm_neon_lib4.o  ./kernel/neon/kernel_strmm_neon_lib4.o ./kernel/neon/kernel_sgemm_strsm_neon_lib4.o ./kernel/neon/kernel_ssyrk_spotrf_neon_lib4.o ./kernel/neon/kernel_sgemv_neon_lib4.o ./kernel/neon/kernel_strmv_neon_lib4.o ./kernel/neon/kernel_strsv_neon_lib4.o ./kernel/neon/kernel_ssymv_c99_lib4.o ./kernel/neon/kernel_stran_neon_lib4.o 
BLAS_OBJS = ./blas/blas_d_lib4.o ./blas/blas_s_lib4.o
MPC_OBJS = ./mpc_solvers/d_ip_box.o ./mpc_solvers/d_res_ip_box.o ./mpc_solvers/c99/d_aux_ip_c99_lib4.o ./mpc_solvers/d_ip2_box.o ./mpc_solvers/d_admm_box.o ./mpc_solvers/d_admm_soft.o ./mpc_solvers/s_ip_box.o ./mpc_solvers/s_res_ip_box.o ./mpc_solvers/c99/s_aux_ip_c99_lib4.o ./mpc_solvers/s_ip2_box.o ./mpc_solvers/s_admm_box.o ./mpc_solvers/s_admm_soft.o
endif
LQCP_OBJS = ./lqcp_solvers/d_ric_sv.o ./lqcp_solvers/d_res.o ./lqcp_solvers/s_ric_sv.o ./lqcp_solvers/s_res.o
LQCP_CODEGEN_OBJS = ./codegen/d_ric_sv_codegen.o ./codegen/d_res_codegen.o ./codegen/s_ric_sv_codegen.o  ./codegen/s_res_codegen.o 
INTERFACE_OBJS = ./interfaces/c/c_order_dynamic_mem_interface.o ./interfaces/c/c_order_static_mem_interface.o ./interfaces/c/fortran_order_dynamic_mem_interface.o ./interfaces/c/fortran_order_static_mem_interface.o

all: clean library test_problem run

codegen: clean codegenerator test_problem run

library: target 
	make -C auxiliary obj
	make -C kernel obj
	make -C blas obj
	make -C lqcp_solvers obj
	make -C mpc_solvers obj
	make -C interfaces obj
	ar rcs libhpmpc.a $(AUX_OBJS) $(KERNEL_OBJS_DOUBLE) $(KERNEL_OBJS_SINGLE) $(BLAS_OBJS) $(LQCP_OBJS) $(MPC_OBJS) $(INTERFACE_OBJS)
	@echo
	@echo " libhpmpc.a static library build complete."
	@echo

shared: target
	make -C auxiliary obj
	make -C kernel obj
	make -C blas obj
	make -C lqcp_solvers obj
	make -C mpc_solvers obj
	make -C interfaces obj
	gcc -shared -o libhpmpc.so $(AUX_OBJS) $(KERNEL_OBJS_DOUBLE) $(KERNEL_OBJS_SINGLE) $(BLAS_OBJS) $(LQCP_OBJS) $(MPC_OBJS) $(INTERFACE_OBJS)
	@echo
	@echo " libhpmpc.so shared library build complete."
	@echo

codegenerator: target
	make -C auxiliary obj
	make -C kernel obj
	make -C blas obj
	touch ./codegen/d_ric_sv_codegen.c 
	touch ./codegen/s_ric_sv_codegen.c
	touch ./codegen/d_res_codegen.c 
	touch ./codegen/s_res_codegen.c 
	make -C codegen obj
	make -C mpc_solvers obj
	make -C interfaces obj
	ar rcs libhpmpc.a $(AUX_OBJS) $(KERNEL_OBJS_DOUBLE) $(KERNEL_OBJS_SINGLE) $(BLAS_OBJS) $(LQCP_CODEGEN_OBJS) $(MPC_OBJS) $(INTERFACE_OBJS)
	@echo
	@echo " libhpmpc.a code generator build complete."
	@echo

target:
	touch ./include/target.h
ifeq ($(TARGET), X64_AVX2)
	echo "#ifndef TARGET_X64_AVX2\n#define TARGET_X64_AVX2\n#endif" > ./include/target.h
endif
ifeq ($(TARGET), X64_AVX)
	echo "#ifndef TARGET_X64_AVX\n#define TARGET_X64_AVX\n#endif" > ./include/target.h
endif
ifeq ($(TARGET), X64_SSE3)
	echo "#ifndef TARGET_X64_SSE3\n#define TARGET_X64_SSE3\n#endif" > ./include/target.h
endif
ifeq ($(TARGET), C99_4X4)
	echo "#ifndef TARGET_C99_4X4\n#define TARGET_C99_4X4\n#endif" > ./include/target.h
endif
ifeq ($(TARGET), CORTEX_A15)
	echo "#ifndef TARGET_CORTEX_A15\n#define TARGET_CORTEX_A15\n#endif" > ./include/target.h
endif
ifeq ($(TARGET), CORTEX_A9)
	echo "#ifndef TARGET_CORTEX_A9\n#define TARGET_CORTEX_A9\n#endif" > ./include/target.h
endif
ifeq ($(TARGET), CORTEX_A7)
	echo "#ifndef TARGET_CORTEX_A7\n#define TARGET_CORTEX_A7\n#endif" > ./include/target.h
endif

test_problem:
	cp libhpmpc.a ./test_problems/libhpmpc.a
#	cp libhpmpc.a ./interfaces/octave/libhpmpc.a
	make -C test_problems obj
	@echo
	@echo " Test problem build complete."
	@echo

run:
	./test_problems/test.out

#install: library
install:
	cp -f libhpmpc.a /usr/lib/libhpmpc.a
	mkdir -p /usr/include/hpmpc
	cp -rf ./include/*.h /usr/include/hpmpc
	
install_shared:
	cp -f libhpmpc.so /usr/lib/libhpmpc.so
	mkdir -p /usr/include/hpmpc
	cp -rf ./include/*.h /usr/include/hpmpc
	
uninstall:
	rm /usr/lib/libhpmpc.a
	rm -r /usr/include/hpmpc
	
uninstall_shared:
	rm /usr/lib/libhpmpc.so
	rm -r /usr/include/hpmpc
	
clean:
	make -C auxiliary clean
	make -C kernel clean
	make -C blas clean
	make -C lqcp_solvers clean
	make -C mpc_solvers clean
	make -C codegen clean
	make -C test_problems clean
	make -C interfaces/octave clean
	make -C interfaces/c clean
#	rm -f $(OBJS)
#	rm -f test.out
#	rm -f *.s
	rm -f target_generator.out
	rm -f *.o
	rm -f libhpmpc.a
	rm -f libhpmpc.so
#	rm -f ./matlab/HPMPC.a

