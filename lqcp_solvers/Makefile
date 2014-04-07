include ../Makefile.rule

ifeq ($(TARGET), AVX)
OBJS = dricposv.o
CFLAGS = $(OPT) -std=c99 -mavx -fPIC -DTARGET_AVX $(DEBUG)
endif
ifeq ($(TARGET), C99_4X4)
OBJS = dricposv.o
CFLAGS = $(OPT) -std=c99 -fPIC -DTARGET_C99_4X4 $(DEBUG)
endif

obj: $(OBJS)

clean:
	rm -f *.o

