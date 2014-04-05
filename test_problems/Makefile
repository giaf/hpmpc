include ../Makefile.rule

LIBS = -lm #-lopenblas

#OBJS_TEST = test_dgemm_orig.o
#OBJS_TEST = test_dgemm.o
OBJS_TEST = tools.o test_dricposv.o

ifeq ($(TARGET), AVX)
CFLAGS = $(OPT) -std=c99 -mavx -DTARGET_AVX $(DEBUG)
endif

obj: $(OBJS_TEST)
	$(CC) -o test.out $(OBJS_TEST) -L. HPMPC.a $(LIBS) #-pg

clean:
	rm -f *.o
	rm -f test.out
	rm -f HPMPC.a

