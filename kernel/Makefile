include ../Makefile.rule

ifeq ($(TARGET), AVX)
OBJS = kernel_dgemm_avx_lib4.o kernel_dpotrf_sse_lib4.o
CFLAGS = $(OPT) -std=c99 -mavx -fPIC -DTARGET_AVX $(DEBUG)
endif

obj: $(OBJS)

clean:
	rm -f *.o

