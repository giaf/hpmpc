include ./Makefile.rule

ifeq ($(TARGET), AVX)
AUX_OBJS = ./auxiliary/aux_d_c99.o ./auxiliary/block_size_avx.o 
KERNEL_OBJS = ./kernel/kernel_dgemm_avx_lib4.o ./kernel/kernel_dpotrf_sse_lib4.o ./kernel/kernel_dgemv_avx_lib4.o ./kernel/corner_dtrmm_avx_lib4.o ./kernel/corner_dpotrf_sse_lib4.o
BLAS_OBJS = ./blas/blas_d_avx_lib4.o
LQCP_OBJS = ./lqcp_solvers/dricposv.o
LQCP_CODEGEN_OBJS = ./codegen/dricposv_codegen.o
MPC_OBJS = #./mpc_solvers/ip_d_box.o
CFLAGS = $(OPT) -std=c99 -mavx -DTARGET_AVX $(DEBUG)
endif
ifeq ($(TARGET), C99_4X4)
AUX_OBJS = ./auxiliary/aux_d_c99.o ./auxiliary/block_size_c99_4x4.o 
KERNEL_OBJS = ./kernel/kernel_dgemm_c99_lib4.o ./kernel/kernel_dpotrf_c99_lib4.o ./kernel/kernel_dgemv_c99_lib4.o ./kernel/corner_dtrmm_c99_lib4.o ./kernel/corner_dpotrf_c99_lib4.o
BLAS_OBJS = ./blas/blas_d_c99_lib4.o
LQCP_OBJS = ./lqcp_solvers/dricposv.o
LQCP_CODEGEN_OBJS = ./codegen/dricposv_codegen.o
MPC_OBJS = #./mpc_solvers/ip_d_box.o
CFLAGS = $(OPT) -std=c99 -mavx -DTARGET_AVX $(DEBUG)
endif

all: clean library test_problem run

code: clean codegenerator test_problem run

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

codegenerator:
	make -C auxiliary obj
	make -C kernel obj
#	make -C blas obj
	make -C codegen obj
#	make -C mpc_solvers obj
	ar rcs HPMPC.a $(AUX_OBJS) $(KERNEL_OBJS) $(LQCP_CODEGEN_OBJS) $(MPC_OBJS)
	@echo
	@echo " HPMPC.a code generator build complete."
	@echo

test_problem:
	cp HPMPC.a ./test_problems/HPMPC.a
#	cp HPMPC.a ./matlab/HPMPC.a
	make -C test_problems obj
	@echo
	@echo " Test problem build complete."
	@echo

run:
	./test_problems/test.out

clean:
	make -C auxiliary clean
	make -C kernel clean
	make -C blas clean
	make -C lqcp_solvers clean
#	make -C mpc_solvers clean
	make -C codegen clean
	make -C test_problems clean
#	make -C matlab clean
	rm -f $(OBJS)
	rm -f test.out
	rm -f *.s
	rm -f *.o
	rm -f HPMPC.a
#	rm -f ./matlab/HPMPC.a

