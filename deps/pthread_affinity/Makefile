CC ?= gcc
HAVE_PTHREAD_AFFINITY := $(shell $(CC) -D_GNU_SOURCE -x c check.c -o /dev/null -fsyntax-only 2>/dev/null && echo 1 || echo 0)

results.mk: Makefile
	touch results.mk
	@if [ "$(HAVE_PTHREAD_AFFINITY)" = "1" ]; then echo 'LIB_CFLAGS += -DHAVE_PTHREAD_SETAFFINITY=1' >> results.mk; fi
