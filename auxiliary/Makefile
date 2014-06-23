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

ifeq ($(TARGET), X64_AVX)
OBJS = aux_d_c99_lib4.o aux_s_c99_lib4.o block_size_x64_avx.o
endif
ifeq ($(TARGET), X64_SSE3)
OBJS = aux_d_c99_lib4.o aux_s_c99_lib4.o block_size_x64_sse.o
endif
ifeq ($(TARGET), AMD_SSE3)
OBJS = aux_d_c99_lib4.o aux_s_c99_lib4.o block_size_x64_sse.o
endif
ifeq ($(TARGET), NEON)
OBJS = aux_d_c99_lib4.o aux_s_c99_lib4.o block_size_neon.o
endif
ifeq ($(TARGET), POWERPC_G2)
OBJS = aux_d_c99_lib4.o aux_s_c99_lib4.o block_size_ppc.o
endif
ifeq ($(TARGET), C99_4X4)
OBJS = aux_d_c99_lib4.o aux_s_c99_lib4.o block_size_c99_4x4.o
endif
ifeq ($(TARGET), C99_2X2)
OBJS = aux_d_c99_lib2.o aux_s_c99_lib2.o block_size_c99_2x2.o
endif
ifeq ($(TARGET), X86_ATOM)
OBJS = aux_d_c99_lib2.o aux_s_c99_lib2.o block_size_x86_atom.o
endif

obj: $(OBJS)

clean:
	rm -f *.o
