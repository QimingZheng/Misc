all:Thread

CC=g++
CPPFLAGS=-Wall -std=c++11 -ggdb
LDFLAGS=-pthread

Thread:xx.o
	    $(CC) $(LDFLAGS) -o $@ $^

xx.o: 7.cpp
	    $(CC) $(CPPFLAGS) -o $@ -c $^


.PHONY:
	    clean

clean:
	    rm xx.o Thread
