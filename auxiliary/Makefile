include ../Makefile.rule

ifeq ($(TARGET), AVX)
OBJS = aux_d_c99.o block_size_avx.o
CFLAGS = $(OPT) -std=c99 -mavx -fPIC -DTARGET_AVX $(DEBUG)
endif
ifeq ($(TARGET), C99_4X4)
OBJS = aux_d_c99.o block_size_c99_4x4.o
CFLAGS = $(OPT) -std=c99 -mavx -fPIC -DTARGET_C99_4X4 $(DEBUG)
endif

obj: $(OBJS)

clean:
	rm -f *.o
