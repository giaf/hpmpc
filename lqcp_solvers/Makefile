include ../Makefile.rule

ifeq ($(TARGET), AVX)
OBJS = dricposv.o
CFLAGS = $(OPT) -std=c99 -mavx -fPIC -DTARGET_AVX $(DEBUG)
endif

obj: $(OBJS)

clean:
	rm -f *.o

