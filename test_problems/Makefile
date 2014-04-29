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

LIBS = -lm #-lopenblas

#OBJS_TEST = test_dgemm_orig.o
#OBJS_TEST = test_sgemm_orig.o
#OBJS_TEST = test_blas_d.o
#OBJS_TEST = test_blas_s.o
#OBJS_TEST = tools.o test_dricposv.o
OBJS_TEST = tools.o test_sricposv.o
#OBJS_TEST = tools.o test_ip_box_lib.o

ifeq ($(TARGET), X64_AVX)
CFLAGS = $(OPT) -std=c99 -mavx -DTARGET_X64_AVX $(DEBUG)
endif
ifeq ($(TARGET), X64_SSE3)
CFLAGS = $(OPT) -std=c99 -msse3 -DTARGET_X64_SSE3 $(DEBUG)
endif
ifeq ($(TARGET), NEON)
CFLAGS = $(OPT) -std=c99 -marm -mfloat-abi=softfp -mfpu=neon -mcpu=cortex-a9 -DTARGET_NEON $(DEBUG)
endif
ifeq ($(TARGET), POWERPC_G2)
CFLAGS = $(OPT) -std=c99 -mcpu=603e -DTARGET_POWERPC_G2 $(DEBUG)
endif
ifeq ($(TARGET), C99_4X4)
CFLAGS = $(OPT) -std=c99 -DTARGET_C99_4X4 $(DEBUG)
endif
ifeq ($(TARGET), C99_2X2)
CFLAGS = $(OPT) -std=c99 -DTARGET_C99_2X2 $(DEBUG)
endif
ifeq ($(TARGET), X86_ATOM)
CFLAGS = $(OPT) -std=c99 -msse3 -mfpmath=sse -march=atom -DTARGET_X86_ATOM $(DEBUG)
endif

obj: $(OBJS_TEST)
	$(CC) -o test.out $(OBJS_TEST) -L. HPMPC.a $(LIBS) #-pg

clean:
	rm -f *.o
	rm -f test.out
	rm -f HPMPC.a

test_dricposv.o: test_dricposv.c test_param.h ../problem_size.h
test_sricposv.o: test_sricposv.c test_param.h ../problem_size.h
test_ip_box_lib.o: test_ip_box_lib.c test_param.h ../problem_size.h
