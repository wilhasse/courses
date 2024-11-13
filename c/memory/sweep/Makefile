# Compiler settings
CC = gcc
CFLAGS = -Wall -Wextra -Wno-unused-but-set-parameter -g -I../munit -I../bootlib

# Source files
LOCAL_SOURCES = vm.c stack.c sneknew.c snekobject.c main.c
MUNIT_SOURCES = ../munit/munit.c
BOOT_SOURCES  = ../bootlib/bootlib.c
SOURCES = $(LOCAL_SOURCES) $(MUNIT_SOURCES) $(BOOT_SOURCES)

# Headers
LOCAL_HEADERS = vm.h stack.h sneknew.h snekobject.h
MUNIT_HEADERS = ../munit/munit.h
BOOT_HEADERS = ../bootlib/bootlib.h
HEADERS = $(LOCAL_HEADERS) $(MUNIT_HEADERS) $(BOOT_HEADERS)

OBJECTS = $(SOURCES:.c=.o)
TARGET = test_snekobject

# Default target
all: $(TARGET)

# Link the final executable
$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET) -lm

# Compile source files to object files
%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

# Special rule for munit.o since it's in a different directory
../munit/munit.o: ../munit/munit.c ../munit/munit.h
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build files
clean:
	rm -f $(OBJECTS) $(TARGET)
	rm -f ../munit/*.o

# Run the tests
test: $(TARGET)
	./$(TARGET)

# Phony targets
