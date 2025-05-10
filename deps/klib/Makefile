results.mk:
	wget https://github.com/attractivechaos/klib/raw/refs/heads/master/khash.h
	wget https://github.com/attractivechaos/klib/raw/refs/heads/master/kbtree.h
	wget https://github.com/attractivechaos/klib/raw/refs/heads/master/ksort.h
	wget https://github.com/attractivechaos/klib/raw/refs/heads/master/kvec.h
	echo "LIB_CFLAGS += -I$(CURDIR)" >> results.mk
	touch $@
