include ../Makefile.rule

ifeq ($(TARGET), AVX)
OBJS = kernel_dgemm_avx_lib4.o kernel_dpotrf_sse_lib4.o kernel_dgemv_avx_lib4.o corner_dtrmm_avx_lib4.o corner_dpotrf_sse_lib4.o
CFLAGS = $(OPT) -std=c99 -mavx -fPIC -DTARGET_AVX $(DEBUG)
endif
ifeq ($(TARGET), C99_4X4)
OBJS = kernel_dgemm_c99_lib4.o kernel_dpotrf_c99_lib4.o kernel_dgemv_c99_lib4.o corner_dtrmm_c99_lib4.o corner_dpotrf_c99_lib4.o
CFLAGS = $(OPT) -std=c99 -fPIC -DTARGET_C99_4X4 $(DEBUG)
endif

obj: $(OBJS)

clean:
	rm -f *.o

