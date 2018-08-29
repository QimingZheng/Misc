CC = gcc

CFLAGS = -g -Wall

LDFLAGS = -lpthread



all: proxy

csapp.o: csapp.c csapp.h
	$(CC) $(CFLAGS) -c csapp.c

proxy.o: proxy.c csapp.h
	$(CC) $(CFLAGS) -c proxy.c

proxy: proxy.o csapp.o
	$(CC) $(CFLAGS) proxy.o csapp.o -o proxy $(LDFLAGS)


clean:
	rm -f *~ *.o proxy core *.tar *.zip *.gzip *.bzip *.gz
