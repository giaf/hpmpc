include ../Makefile.rule

ifeq ($(TARGET), AVX)
OBJS = aux_d_c99.o block_size_avx.o
CFLAGS = $(OPT) -std=c99 -mavx -fPIC -DTARGET_AVX $(DEBUG)
endif

obj: $(OBJS)

clean:
	rm -f *.o
