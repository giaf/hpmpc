include ./Makefile.rule

ifeq ($(TARGET), AVX)
AUX_OBJS = ./auxiliary/aux_d_c99.o ./auxiliary/block_size_avx.o 
KERNEL_OBJS = ./kernel/kernel_dgemm_avx_lib4.o ./kernel/kernel_dpotrf_sse_lib4.o ./kernel/kernel_dgemv_avx_lib4.o 
BLAS_OBJS = ./blas/blas_d_avx_lib4.o
LQCP_OBJS = ./lqcp_solvers/dricposv.o
MPC_OBJS = #./mpc_solvers/ip_d_box.o
CFLAGS = $(OPT) -std=c99 -mavx -DTARGET_AVX $(DEBUG)
endif

library:
	make -C auxiliary obj
	make -C kernel obj
	make -C blas obj
	make -C lqcp_solvers obj
#	make -C mpc_solvers obj
	ar rcs HPMPC.a $(AUX_OBJS) $(KERNEL_OBJS) $(BLAS_OBJS) $(LQCP_OBJS) $(MPC_OBJS)
	@echo
	@echo " HPMPC.a library build complete."
	@echo

test: library
	cp HPMPC.a ./test/HPMPC.a
#	cp HPMPC.a ./matlab/HPMPC.a
	make -C test obj
	@echo
	@echo " Test problem build complete."
	@echo

run:
	./test/test.out

clean:
	make -C auxiliary clean
	make -C kernel clean
	make -C blas clean
	make -C lqcp_solvers clean
#	make -C mpc_solvers clean
	make -C test clean
#	make -C matlab clean
	rm -f $(OBJS)
	rm -f test.out
	rm -f *.s
	rm -f *.o
	rm -f HPMPC.a
#	rm -f ./matlab/HPMPC.a

