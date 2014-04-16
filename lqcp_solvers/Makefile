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
OBJS = dricposv.o
CFLAGS = $(OPT) -std=c99 -mavx -fPIC -DTARGET_AVX $(DEBUG)
endif
ifeq ($(TARGET), SSE4)
OBJS = dricposv.o
CFLAGS = $(OPT) -std=c99 -msse4 -fPIC -DTARGET_SSE4 $(DEBUG)
endif
ifeq ($(TARGET), NEON)
OBJS = dricposv.o sricposv.o
CFLAGS = $(OPT) -std=c99 -fPIC -marm -mfloat-abi=softfp -mfpu=neon -mcpu=cortex-a9 -DTARGET_NEON $(DEBUG)
endif
ifeq ($(TARGET), POWERPC_G2)
OBJS = dricposv.o sricposv.o
CFLAGS = $(OPT) -std=c99 -fPIC -fPIC -mcpu=603e -DTARGET_POWERPC_G2 $(DEBUG)
endif
ifeq ($(TARGET), C99_4X4)
OBJS = dricposv.o sricposv.o
CFLAGS = $(OPT) -std=c99 -fPIC -DTARGET_C99_4X4 $(DEBUG)
endif
ifeq ($(TARGET), C99_2X2)
OBJS = dricposv.o sricposv.o
CFLAGS = $(OPT) -std=c99 -fPIC -DTARGET_C99_2X2 $(DEBUG)
endif
ifeq ($(TARGET), ATOM)
OBJS = dricposv.o sricposv.o
CFLAGS = $(OPT) -std=c99 -msse3 -mfpmath=sse -march=atom -fPIC -DTARGET_ATOM $(DEBUG)
endif

obj: $(OBJS)

clean:
	rm -f *.o

