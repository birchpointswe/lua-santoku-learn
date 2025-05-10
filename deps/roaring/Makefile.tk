results.mk:
	wget https://github.com/RoaringBitmap/CRoaring/releases/download/v4.3.1/roaring.c
	wget https://github.com/RoaringBitmap/CRoaring/releases/download/v4.3.1/roaring.h
	echo "LIB_CFLAGS += -isystem $(CURDIR)" >> results.mk
	touch $@
