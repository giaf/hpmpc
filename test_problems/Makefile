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

#OBJS_TEST = test_blas_d_orig.o
#OBJS_TEST = test_blas_s_orig.o
#OBJS_TEST = test_blas_d.o
OBJS_TEST = test_blas_s.o
#OBJS_TEST = tools.o test_d_ric_sv.o
#OBJS_TEST = tools.o test_s_ric_sv.o
#OBJS_TEST = tools.o test_d_ip_box.o
#OBJS_TEST = tools.o test_s_ip_box.o

obj: $(OBJS_TEST)
	$(CC) -o test.out $(OBJS_TEST) -L. libhpmpc_pro.a $(LIBS) #-pg

clean:
	rm -f *.o
	rm -f test.out
	rm -f libhpmpc.a

test_blas_d_origin.o: test_blas_d_origin.c test_param.h ../problem_size.h
test_d_ric_sv.o: test_d_ric_sv.c test_param.h ../problem_size.h
test_s_ric_sv.o: test_s_ric_sv.c test_param.h ../problem_size.h
test_d_ip_box.o: test_d_ip_box.c test_param.h ../problem_size.h
test_s_ip_box.o: test_s_ip_box.c test_param.h ../problem_size.h
