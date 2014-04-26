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

ifeq ($(TARGET), AVX)
OBJS = blas_d_lib4.o blas_s_lib4.o
CFLAGS = $(OPT) -std=c99 -mavx -DTARGET_AVX -fPIC $(DEBUG)
endif
ifeq ($(TARGET), X64_SSE3)
OBJS = blas_d_lib4.o blas_s_lib4.o
CFLAGS = $(OPT) -std=c99 -msse3 -DTARGET_X64_SSE3 -fPIC $(DEBUG)
endif
ifeq ($(TARGET), NEON)
OBJS = blas_d_lib4.o blas_s_lib4.o
CFLAGS = $(OPT) -std=c99 -fPIC -marm -mfloat-abi=softfp -mfpu=neon -mcpu=cortex-a9 -DTARGET_NEON $(DEBUG)
endif
ifeq ($(TARGET), POWERPC_G2)
OBJS = blas_d_lib4.o blas_s_lib4.o
CFLAGS = $(OPT) -std=c99 -fPIC -mcpu=603e -DTARGET_POWERPC_G2 $(DEBUG)
endif
ifeq ($(TARGET), C99_4X4)
OBJS = blas_d_lib4.o blas_s_lib4.o
CFLAGS = $(OPT) -std=c99 -DTARGET_C99_4X4 -fPIC $(DEBUG)
endif
ifeq ($(TARGET), C99_2X2)
OBJS = blas_d_lib2.o blas_s_lib2.o
CFLAGS = $(OPT) -std=c99 -DTARGET_C99_2X2 -fPIC $(DEBUG)
endif
ifeq ($(TARGET), X86_ATOM)
OBJS = blas_d_lib2.o blas_s_lib4.o
CFLAGS = $(OPT) -std=c99 -msse3 -mfpmath=sse -march=atom -DTARGET_X86_ATOM -fPIC $(DEBUG)
endif

obj: $(OBJS)

clean:
	rm -f *.o

