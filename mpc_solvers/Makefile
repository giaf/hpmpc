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

ifeq ($(TARGET), X64_AVX2)
OBJS = d_ip_box.o d_res_ip_box.o d_aux_ip_avx_lib4.o d_ip2_box.o #s_ip_box.o s_ip2_box.o sres_ip_box.o 
endif
ifeq ($(TARGET), X64_AVX)
OBJS = d_ip_box.o d_res_ip_box.o d_aux_ip_avx_lib4.o d_ip2_box.o #s_ip_box.o s_ip2_box.o sres_ip_box.o 
endif
ifeq ($(TARGET), X64_SSE3)
OBJS = d_ip_box.o d_res_ip_box.o d_aux_ip_c99_lib4.o d_ip2_box.o s_ip_box.o s_res_ip_box.o s_aux_ip_c99_lib4.o s_ip2_box.o
endif
ifeq ($(TARGET), C99_4X4)
OBJS = d_ip_box.o d_res_ip_box.o d_aux_ip_c99_lib4.o d_ip2_box.o s_ip_box.o s_res_ip_box.o s_aux_ip_c99_lib4.o s_ip2_box.o
endif
ifeq ($(TARGET), CORTEX_A15)
OBJS = d_ip_box.o d_res_ip_box.o d_aux_ip_c99_lib4.o d_ip2_box.o s_ip_box.o s_res_ip_box.o s_aux_ip_c99_lib4.o s_ip2_box.o
endif
ifeq ($(TARGET), CORTEX_A9)
OBJS = d_ip_box.o d_res_ip_box.o d_aux_ip_c99_lib4.o d_ip2_box.o s_ip_box.o s_res_ip_box.o s_aux_ip_c99_lib4.o s_ip2_box.o
endif

obj: $(OBJS)

clean:
	rm -f *.o


