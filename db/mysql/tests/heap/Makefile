CC = g++
CFLAGS = -Wall -g -std=c++11
SOURCES = hp_block.cc hp_clear.cc hp_create.cc hp_delete.cc hp_dspace.cc \
          hp_extra.cc hp_hash.cc hp_info.cc hp_open.cc hp_panic.cc \
          hp_record.cc hp_scan.cc hp_static.cc hp_update.cc hp_write.cc \
          heap_example.cc

INCLUDES = -I./include

OBJECTS = $(SOURCES:.cc=.o)
TARGET = heap_example

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET)

.cc.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $<

clean:
	rm -f $(OBJECTS) $(TARGET)
