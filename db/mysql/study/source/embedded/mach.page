# mach0data.c

Below is a step-by-step explanation of what these two functions do and how InnoDB’s “compressed integer” reading works in this particular snippet. The code is designed to parse variable-length integers from a byte stream (e.g., from the InnoDB log) in a memory-efficient format.

---

## High-Level Overview

1. **`mach_parse_compressed()`**  
   - Reads a **32-bit** integer (`ulint`) from a buffer in a variable-length (compressed) format.
   - Depending on the high-order bits of the first byte, the integer might occupy between **1 and 5 bytes**.
   - If the code cannot fully read the required bytes (i.e., the buffer ends prematurely), the function returns `NULL`.

2. **`mach_dulint_parse_compressed()`**  
   - Reads a **64-bit** integer (`dulint`) from a buffer in a somewhat “split” compressed format:
     - First part is read via a compressed scheme (the “high” 32 bits).
     - Second part is read as a fixed 4-byte integer (the “low” 32 bits).
   - If there aren’t enough bytes in the buffer, it also returns `NULL`.

Both functions return a pointer to the position **after** the parsed data on success, or `NULL` on failure (insufficient bytes in the buffer).

---

## `mach_parse_compressed()`: Reading a 32-bit Compressed Integer

```c
byte*
mach_parse_compressed(
    byte*   ptr,
    byte*   end_ptr,
    ulint*  val)
{
    ulint   flag;

    ut_ad(ptr && end_ptr && val);

    /* Check if there is at least one byte to read */
    if (ptr >= end_ptr) {
        return(NULL);
    }

    /* Read the first byte, which is a "flag" that indicates length */
    flag = mach_read_from_1(ptr);

    /* Case 1: Single byte value (< 0x80) */
    if (flag < 0x80UL) {
        *val = flag;
        return(ptr + 1);

    /* Case 2: Two-byte value (0x80 <= flag < 0xC0) */
    } else if (flag < 0xC0UL) {
        if (end_ptr < ptr + 2) {
            return(NULL);
        }
        /* Extract the lower 15 bits */
        *val = mach_read_from_2(ptr) & 0x7FFFUL;
        return(ptr + 2);

    /* Case 3: Three-byte value (0xC0 <= flag < 0xE0) */
    } else if (flag < 0xE0UL) {
        if (end_ptr < ptr + 3) {
            return(NULL);
        }
        /* Extract the lower 22 bits */
        *val = mach_read_from_3(ptr) & 0x3FFFFFUL;
        return(ptr + 3);

    /* Case 4: Four-byte value (0xE0 <= flag < 0xF0) */
    } else if (flag < 0xF0UL) {
        if (end_ptr < ptr + 4) {
            return(NULL);
        }
        /* Extract the lower 29 bits */
        *val = mach_read_from_4(ptr) & 0x1FFFFFFFUL;
        return(ptr + 4);

    /* Case 5: Five-byte value (exactly 0xF0) */
    } else {
        ut_ad(flag == 0xF0UL);

        if (end_ptr < ptr + 5) {
            return(NULL);
        }
        /* The next four bytes hold the full 32-bit value */
        *val = mach_read_from_4(ptr + 1);
        return(ptr + 5);
    }
}
```

### How It Works Step by Step

1. **Check Buffer Availability**  
   - If the current pointer `ptr` is already at or beyond `end_ptr`, we don’t have even one byte to read, so return `NULL` immediately.

2. **Read the First Byte**  
   - Store the first byte in `flag`. The bits in `flag` determine how large the stored integer actually is.

3. **Interpret the Flag**  
   - If `flag < 0x80`, the integer is stored in **just that one byte**. So `val = flag`.
   - If `0x80 <= flag < 0xC0`, the integer is stored in **two bytes**.  
     - We use `mach_read_from_2(ptr)` and then mask off bits we don’t need (`& 0x7FFFUL`).
   - If `0xC0 <= flag < 0xE0`, the integer takes **three bytes** in total.  
     - We read them with `mach_read_from_3(ptr)` and mask off bits with `& 0x3FFFFFUL`.
   - If `0xE0 <= flag < 0xF0`, the integer is **four bytes**.  
     - We do `mach_read_from_4(ptr)` and mask off `& 0x1FFFFFFFUL`.
   - If the value is exactly `0xF0`, it indicates the integer occupies **5 bytes**:  
     - The first byte is the flag (0xF0), and the **next four bytes** contain the actual 32-bit integer.

4. **Bounds Checking**  
   - Before attempting to read additional bytes (2, 3, 4, or 5 total), the code checks `end_ptr < ptr + N`. If not enough space remains, it returns `NULL`.

5. **Return**  
   - On success, it returns the pointer to the position immediately **after** the last byte of the integer.  
   - On failure (insufficient bytes), it returns `NULL`.

This is basically a variable-length encoding for **up to** 32 bits, where the leading bits of the first byte signal how many total bytes are used to store the integer.

---

## `mach_dulint_parse_compressed()`: Reading a 64-bit Compressed Integer (`dulint`)

```c
byte*
mach_dulint_parse_compressed(
    byte*   ptr,
    byte*   end_ptr,
    dulint* val)
{
    ulint   high;
    ulint   low;
    ulint   size;

    ut_ad(ptr && end_ptr && val);

    if (end_ptr < ptr + 5) {
        return(NULL);
    }

    /* 1) Read the first portion (the "high" 32 bits) in compressed form */
    high = mach_read_compressed(ptr);
    size = mach_get_compressed_size(high);

    ptr += size;

    if (end_ptr < ptr + 4) {
        return(NULL);
    }

    /* 2) Read the next 4 bytes (the "low" 32 bits) */
    low = mach_read_from_4(ptr);

    /* 3) Combine them into a 'dulint' (InnoDB 64-bit type) */
    *val = ut_dulint_create(high, low);

    return(ptr + 4);
}
```

### How It Works Step by Step

1. **Check Minimum Buffer Size**  
   - The function needs to read at least **5 bytes**: 
     1. At least 1 byte for the compressed “high” part (but potentially up to 5, depending on the compressed size).  
     2. And then 4 bytes for the “low” part.

2. **Read the “High” 32 Bits in a Compressed Form**  
   - `high = mach_read_compressed(ptr);`  
     - This presumably behaves like `mach_parse_compressed()`, but returns the integer directly (rather than a pointer) and also signals how many bytes it consumed via another method or with help from `mach_get_compressed_size()`.
   - `size = mach_get_compressed_size(high);`  
     - Figures out how many bytes the compressed integer used.  
   - `ptr += size;`  
     - Advance the pointer by that amount.

3. **Read the “Low” 32 Bits in Fixed Format**  
   - After reading the compressed high bits, the next **4 bytes** are simply read with `mach_read_from_4(ptr)`.

4. **Combine to Form a 64-bit Value**  
   - `ut_dulint_create(high, low)` merges these two 32-bit chunks into a single 64-bit integer (`dulint`).
   - Assign it to `*val`.

5. **Return**  
   - Advance the pointer by 4 more bytes (the part that was read for `low`) and return it.  
   - If at any point there aren’t enough bytes in the buffer to complete the read, the function returns `NULL`.

---

## Key Supporting Concepts

1. **`mach_read_from_1(ptr)`**  
   - Reads 1 byte from the pointer and converts it into a `ulint` (typically 32-bit unsigned on most systems).

2. **`mach_read_from_2(ptr)`, `mach_read_from_3(ptr)`, `mach_read_from_4(ptr)`**  
   - Read 2, 3, or 4 bytes and convert them into a host-endian 32-bit integer.  
   - Under the hood, these typically do something like:
     ```c
     #define mach_read_from_2(b) ((ulint) (b[0] << 8 | b[1]))
     // etc. for 3 and 4, carefully handling endianness
     ```

3. **Masking with `& 0x7FFFUL`, `& 0x3FFFFFUL`, etc.**  
   - These masks strip away bits that are used as flags in the first byte. For example, if the first byte is `0x9A`, the top 1–3 bits might indicate “this is stored in 2 or 3 bytes,” so those bits are not part of the actual integer value.

4. **`dulint`**  
   - In InnoDB source code, `dulint` is a custom 64-bit structure or type (depending on the version). Typically, it’s represented by two 32-bit values (`high` and `low`) or a single 64-bit if the compiler/architecture supports it.

5. **Variable-Length Encoding**  
   - This code implements a classic variable-length or “compressed” integer format. The idea is to save space by using fewer bytes for smaller numbers, while still allowing large numbers up to 32 bits (or more, for the “dulint” version).

---

## Summary

- **`mach_parse_compressed()`**: Reads a single 32-bit integer in a variable-length format (1–5 bytes).  
- **`mach_dulint_parse_compressed()`**: Reads a 64-bit integer in two parts—a compressed 32-bit “high” portion followed by a fixed 4-byte “low” portion.

Both functions carefully check whether enough bytes remain in the buffer before attempting to parse. If there isn’t enough space, they return `NULL` to indicate an incomplete read. This is crucial in log parsing and recovery code where partial or corrupted data might be encountered.

Overall, these functions are foundational for reading compactly stored integers in InnoDB’s pages and log records, allowing InnoDB to handle data in a space-efficient manner while still supporting large numeric values.