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

include ../../Makefile.rule

ifeq ($(TARGET), CORTEX_A15)
OBJS_DOUBLE = kernel_dgemm_vfpv3_lib4.o kernel_dtrmm_c99_lib4.o kernel_dtrsm_c99_lib4.o kernel_dpotrf_c99_lib4.o kernel_dgemv_c99_lib4.o kernel_dtrmv_c99_lib4.o kernel_dtrsv_c99_lib4.o kernel_dsymv_c99_lib4.o kernel_dtran_c99_lib4.o
OBJS_SINGLE = #kernel_sgemm_avx_lib8.o #kernel_spotrf_sse_lib4.o kernel_sgemv_sse_lib4.o corner_strmm_sse_lib4.o corner_spotrf_c99_lib4.o kernel_ssymv_sse_lib4.o
CFLAGS = $(OPT) -std=c99 -DTARGET_CORTEX_A15 -fPIC -marm -mfloat-abi=hard -mfpu=neon -mcpu=cortex-a15 $(DEBUG)
endif

obj: $(OBJS_DOUBLE) $(OBJS_SINGLE)

clean:
	rm -f *.o

