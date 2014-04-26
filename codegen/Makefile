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

include ../Makefile.rule

OBJS = dricposv_codegen.o sricposv_codegen.o dres_codegen.o sres_codegen.o

ifeq ($(TARGET), AVX)
CODEGEN_OBJS_DOUBLE = code_generator_blas_d_lib4.o code_generator_dricposv.o
CODEGEN_OBJS_SINGLE = code_generator_blas_s_lib4.o code_generator_sricposv.o
CFLAGS = $(OPT) -std=c99 -mavx -DTARGET_AVX $(DEBUG)
endif
ifeq ($(TARGET), X64_SSE3)
CODEGEN_OBJS_DOUBLE = code_generator_blas_d_lib4.o code_generator_dricposv.o
CODEGEN_OBJS_SINGLE = code_generator_blas_s_lib4.o code_generator_sricposv.o
CFLAGS = $(OPT) -std=c99 -msse3 -DTARGET_X64_SSE3 $(DEBUG)
endif
ifeq ($(TARGET), NEON)
CODEGEN_OBJS_DOUBLE = code_generator_blas_d_lib4.o code_generator_dricposv.o
CODEGEN_OBJS_SINGLE = code_generator_blas_s_lib4.o code_generator_sricposv.o
CFLAGS = $(OPT) -std=c99 -fPIC -marm -mfloat-abi=softfp -mfpu=neon -mcpu=cortex-a9 -DTARGET_NEON $(DEBUG)
endif
ifeq ($(TARGET), POWERPC_G2)
CODEGEN_OBJS_DOUBLE = code_generator_blas_d_lib4.o code_generator_dricposv.o
CODEGEN_OBJS_SINGLE = code_generator_blas_s_lib4.o code_generator_sricposv.o
CFLAGS = $(OPT) -std=c99 -fPIC -mcpu=603e -DTARGET_POWERPC_G2 $(DEBUG)
endif
ifeq ($(TARGET), C99_4X4)
CODEGEN_OBJS_DOUBLE = code_generator_blas_d_lib4.o code_generator_dricposv.o
CODEGEN_OBJS_SINGLE = code_generator_blas_s_lib4.o code_generator_sricposv.o
CFLAGS = $(OPT) -std=c99 -DTARGET_C99_4X4 $(DEBUG)
endif
ifeq ($(TARGET), C99_2X2)
CODEGEN_OBJS_DOUBLE = code_generator_blas_d_lib2.o code_generator_dricposv.o
CODEGEN_OBJS_SINGLE = code_generator_blas_s_lib2.o code_generator_sricposv.o
CFLAGS = $(OPT) -std=c99 -DTARGET_C99_2X2 $(DEBUG)
endif
ifeq ($(TARGET), X86_ATOM)
CODEGEN_OBJS_DOUBLE = code_generator_blas_d_lib2.o code_generator_dricposv.o
CODEGEN_OBJS_SINGLE = code_generator_blas_s_lib4.o code_generator_sricposv.o
CFLAGS = $(OPT) -std=c99 -msse3 -mfpmath=sse -march=atom -DTARGET_X86_ATOM $(DEBUG)
endif

codegen: $(CODEGEN_OBJS_DOUBLE) $(CODEGEN_OBJS_SINGLE)
	gcc -o code_generator.out $(CODEGEN_OBJS_DOUBLE) -O2
	./code_generator.out
	gcc -o code_generator.out $(CODEGEN_OBJS_SINGLE) -O2
	./code_generator.out
#	sleep 5

obj: codegen $(OBJS)

clean:
	rm -f dricposv_codegen.c 
	rm -f sricposv_codegen.c
	rm -f dres_codegen.c 
	rm -f sres_codegen.c
	rm -f *.o
	rm -f code_generator.out

code_generator_dricposv.o: code_generator_dricposv.c ../problem_size.h
code_generator_sricposv.o: code_generator_dricposv.c ../problem_size.h
code_generator_blas_d_lib4.o: code_generator_blas_d_lib4.c ../problem_size.h
code_generator_blas_s_lib4.o: code_generator_blas_s_lib4.c ../problem_size.h
code_generator_blas_d_lib2.o: code_generator_blas_d_lib2.c ../problem_size.h
code_generator_blas_s_lib2.o: code_generator_blas_s_lib2.c ../problem_size.h

