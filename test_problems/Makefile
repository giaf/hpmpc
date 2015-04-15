###################################################################################################
#                                                                                                 #
# This file is part of HPMPC.                                                                     #
#                                                                                                 #
# HPMPC -- Library for High-Performance implementation of solvers for MPC.                        #
# Copyright (C) 2014-2015 by Technical University of Denmark. All rights reserved.                #
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

LIBS = -lm 
#LIBS = -lm -lopenblas -pthread
#LIBS = -lblis -lm -fopenmp

#OBJS_TEST = test_blas_d_debug.o
#OBJS_TEST = test_blas_s_debug.o
OBJS_TEST = test_blas_d.o
#OBJS_TEST = test_blas_s.o
#OBJS_TEST = tools.o test_d_ric_mpc.o
#OBJS_TEST = tools.o test_s_ric_mpc.o
#OBJS_TEST = tools.o test_d_ric_mhe.o
#OBJS_TEST = tools.o test_d_ip_hard.o
#OBJS_TEST = tools.o test_s_ip_box.o
#OBJS_TEST = tools.o test_d_ip_soft.o
#OBJS_TEST = tools.o test_d_admm_box.o
#OBJS_TEST = tools.o test_s_admm_box.o
#OBJS_TEST = tools.o test_d_admm_soft.o
#OBJS_TEST = tools.o test_s_admm_soft.o

obj: $(OBJS_TEST)
	$(CC) -o test.out $(OBJS_TEST) -L. libhpmpc.a $(LIBS) #-pg

clean:
	rm -f *.o
	rm -f test.out
	rm -f libhpmpc.a

test_blas_d_debug.o: test_blas_d_debug.c test_param.h ../problem_size.h
test_blas_s_debug.o: test_blas_s_debug.c test_param.h ../problem_size.h
test_blas_d.o: test_blas_d.c test_param.h ../problem_size.h
test_blas_s.o: test_blas_s.c test_param.h ../problem_size.h
test_d_ric_mpc.o: test_d_ric_mpc.c test_param.h ../problem_size.h
test_s_ric_mpc.o: test_s_ric_mpc.c test_param.h ../problem_size.h
test_d_ric_mhe.o: test_d_ric_mhe.c test_param.h ../problem_size.h
test_d_ip_hard.o: test_d_ip_hard.c test_param.h ../problem_size.h
test_s_ip_box.o: test_s_ip_box.c test_param.h ../problem_size.h
test_d_ip_soft.o: test_d_ip_soft.c test_param.h ../problem_size.h
test_d_admm_box.o: test_d_admm_box.c test_param.h ../problem_size.h
test_s_admm_box.o: test_s_admm_box.c test_param.h ../problem_size.h
test_d_admm_soft.o: test_d_admm_soft.c test_param.h ../problem_size.h
test_s_admm_soft.o: test_s_admm_soft.c test_param.h ../problem_size.h

