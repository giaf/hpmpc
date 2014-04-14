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

include ../Makefile.rule

ifeq ($(TARGET), AVX)
OBJS_DOUBLE = kernel_dgemm_avx_lib4.o kernel_dpotrf_sse_lib4.o kernel_dgemv_avx_lib4.o corner_dtrmm_avx_lib4.o corner_dpotrf_sse_lib4.o
CFLAGS = $(OPT) -std=c99 -mavx -fPIC -DTARGET_AVX $(DEBUG)
endif
ifeq ($(TARGET), NEON)
OBJS_DOUBLE = kernel_dgemm_neon_lib4.o kernel_dpotrf_c99_lib4.o kernel_dgemv_c99_lib4.o corner_dtrmm_c99_lib4.o corner_dpotrf_c99_lib4.o
OBJS_SINGLE = kernel_sgemm_neon_lib4.o kernel_spotrf_c99_lib4.o kernel_sgemv_c99_lib4.o corner_strmm_c99_lib4.o corner_spotrf_c99_lib4.o
CFLAGS = $(OPT) -std=c99 -fPIC -marm -mfloat-abi=softfp -mfpu=neon -mcpu=cortex-a9 -DTARGET_NEON $(DEBUG)
endif
ifeq ($(TARGET), POWERPC_G2)
OBJS_DOUBLE = kernel_dgemm_ppc_g2_lib4.o kernel_dpotrf_c99_lib4.o kernel_dgemv_c99_lib4.o corner_dtrmm_c99_lib4.o corner_dpotrf_c99_lib4.o
OBJS_SINGLE = kernel_sgemm_ppc_g2_lib4.o kernel_spotrf_c99_lib4.o kernel_sgemv_c99_lib4.o corner_strmm_c99_lib4.o corner_spotrf_c99_lib4.o
CFLAGS = $(OPT) -std=c99 -fPIC -mcpu=603e -DTARGET_POWERPC_G2 $(DEBUG)
endif
ifeq ($(TARGET), C99_4X4)
OBJS_DOUBLE = kernel_dgemm_c99_lib4.o kernel_dpotrf_c99_lib4.o kernel_dgemv_c99_lib4.o corner_dtrmm_c99_lib4.o corner_dpotrf_c99_lib4.o 
OBJS_SINGLE = kernel_sgemm_c99_lib4.o kernel_spotrf_c99_lib4.o kernel_sgemv_c99_lib4.o corner_strmm_c99_lib4.o corner_spotrf_c99_lib4.o
CFLAGS = $(OPT) -std=c99 -fPIC -DTARGET_C99_4X4 $(DEBUG)
endif
ifeq ($(TARGET), C99_2X2)
OBJS_DOUBLE = kernel_dgemm_c99_lib2.o kernel_dpotrf_c99_lib2.o kernel_dgemv_c99_lib2.o corner_dtrmm_c99_lib2.o corner_dpotrf_c99_lib2.o
OBJS_SINGLE = kernel_sgemm_c99_lib2.o kernel_spotrf_c99_lib2.o kernel_sgemv_c99_lib2.o corner_strmm_c99_lib2.o corner_spotrf_c99_lib2.o
CFLAGS = $(OPT) -std=c99 -fPIC -DTARGET_C99_2X2 $(DEBUG)
endif
ifeq ($(TARGET), ATOM)
OBJS_DOUBLE = kernel_dgemm_atom_lib2.o kernel_dpotrf_c99_lib2.o kernel_dgemv_c99_lib2.o corner_dtrmm_c99_lib2.o corner_dpotrf_c99_lib2.o
OBJS_SINGLE = kernel_sgemm_atom_lib4.o kernel_spotrf_c99_lib4.o kernel_sgemv_atom_lib4.o corner_strmm_c99_lib4.o corner_spotrf_c99_lib4.o
CFLAGS = $(OPT) -std=c99 -msse3 -mfpmath=sse -march=atom -fPIC -DTARGET_ATOM $(DEBUG)
endif

obj: $(OBJS_DOUBLE) $(OBJS_SINGLE)

clean:
	rm -f *.o

