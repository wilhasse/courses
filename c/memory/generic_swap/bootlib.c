// bootlib.c
#include "bootlib.h"
#include <stdio.h>

// Undefine the macros so we can use the real malloc/free inside our implementation
#undef malloc
#undef free

#define MAX_ALLOCATIONS 1000

static struct {
    struct {
        void* ptr;
        const char* file;
        int line;
    } allocations[MAX_ALLOCATIONS];
    size_t count;
} tracker = {0};

void* tracked_malloc(size_t size, const char* file, int line) {
    // Use the real malloc here, not the macro
    void* ptr = malloc(size);
    if (ptr && tracker.count < MAX_ALLOCATIONS) {
        tracker.allocations[tracker.count].ptr = ptr;
        tracker.allocations[tracker.count].file = file;
        tracker.allocations[tracker.count].line = line;
        tracker.count++;
    }
    return ptr;
}

void tracked_free(void* ptr, const char* file, int line) {
    if (!ptr) return;

    for (size_t i = 0; i < tracker.count; i++) {
        if (tracker.allocations[i].ptr == ptr) {
            tracker.allocations[i] = tracker.allocations[--tracker.count];
            // Use the real free here, not the macro
            free(ptr);
            return;
        }
    }

    printf("Warning: Attempting to free untracked pointer at %s:%d\n", file, line);
    free(ptr);
}

bool boot_all_freed(void) {
    if (tracker.count > 0) {
        printf("Memory leak detected! %zu allocations not freed:\n", tracker.count);
        for (size_t i = 0; i < tracker.count; i++) {
            printf("  Leaked memory at %p, allocated at %s:%d\n",
                   tracker.allocations[i].ptr,
                   tracker.allocations[i].file,
                   tracker.allocations[i].line);
        }
    }
    return tracker.count == 0;
}
