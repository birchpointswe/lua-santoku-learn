CC ?= gcc
HAVE_NUMA := $(shell $(CC) -x c check.c -lnuma -o /dev/null -fsyntax-only 2>/dev/null && echo 1 || echo 0)

results.mk: Makefile
	touch results.mk
	@if [ "$(HAVE_NUMA)" = "1" ]; then echo 'LIB_LDFLAGS += -lnuma' >> results.mk; fi
