CC = gcc
CFLAGS = -Wall -Wextra -g -I../munit
OBJS = main.o munit.o

all: program

program: $(OBJS)
	$(CC) $(OBJS) -o program -lm

main.o: main.c exercise.h ../munit/munit.h
	$(CC) $(CFLAGS) -c main.c

munit.o: ../munit/munit.c ../munit/munit.h
	$(CC) $(CFLAGS) -c ../munit/munit.c

clean:
	rm -f program $(OBJS)
