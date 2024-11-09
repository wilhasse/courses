#include <string.h>
#include "exercise.h"

int smart_append(TextBuffer* dest, const char* src) {
    // 1. Check for NULL inputs
    if (dest == NULL || src == NULL) {
        return 1;  // false
    }

    // 2. Define max buffer size
    const size_t MAX_BUFFER_SIZE = 64;

    // 3. Get source string length
    size_t src_len = strlen(src);

    // 4. Calculate remaining space
    // Note: We need space for null terminator, so subtract 1 from MAX_BUFFER_SIZE
    size_t remaining_space = MAX_BUFFER_SIZE - dest->length - 1;

    // 5. Handle case where src is larger than remaining space
    if (src_len > remaining_space) {
        strncat(dest->buffer, src, remaining_space);
        dest->length = MAX_BUFFER_SIZE - 1;  // -1 for null terminator
        return 1;  // false - couldn't append everything
    }

    // 6. Handle case where there's enough space
    strcat(dest->buffer, src);
    dest->length += src_len;
    return 0;  // true - full append was possible
}
