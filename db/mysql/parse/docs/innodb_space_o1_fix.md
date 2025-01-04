Below is an **example approach** to fixing the record-offset logic so that **any number of columns** (including multiple primary-key columns) is handled properly. The core issue is that the original code “hard-coded” offset handling for exactly one user column plus two hidden columns (`trx_id` and `roll_ptr`). InnoDB tables with **composite primary keys** (i.e., multiple columns in the PK) break that assumption.

> **Note:** The exact fix may vary depending on how your JSON SDI file is structured, how you want to display hidden columns, etc. The example below shows one clean way to parse *all* columns dynamically. You can adapt it to your existing code style.

---

## 1. Revamp `rec_init_offsets()`

Instead of hardcoding for a single column in `dict_cols[3]` and then skipping to `[5]` or `[6]` for the hidden columns, we’ll build **all user columns** in a loop. Then, at the end, we append the two hidden columns (`trx_id` and `roll_ptr`).

```cpp
int rec_init_offsets() {
  
  ...
  
  // 2) Number of user columns (from JSON).
  //    Depending on your SDI file format, you may need to adapt indexing:
  auto columns = d[1]["object"]["dd_object"]["columns"];
  const size_t n_user_cols = columns.Size();

  // Store in offsets_[1] how many *user* columns we have
  offsets_[0] = REC_OFFS_NORMAL_SIZE;   // Usually just a “magic” constant
  offsets_[1] = n_user_cols;            // total user columns

  // 3) We track the “running offset” for each column within the record
  ulint current_offset = 0;

  // We'll fill dict_cols[] starting from index=3, so as not to stomp on offsets_[0..2].
  // This is just a convention inherited from the original code.
  ulint dict_idx = 3;

  // 4) Parse each user column from the JSON
  for (size_t i = 0; i < n_user_cols; i++) {
    dict_cols[dict_idx].col_name 
      = columns[i]["name"].GetString();

    dict_cols[dict_idx].column_type_utf8 
      = columns[i]["column_type_utf8"].GetString();

    // Determine length for char(N) vs int, etc.
    int col_len = 0;
    if (dict_cols[dict_idx].column_type_utf8 == "int") {
      col_len = 4;  // typical
    } else if (dict_cols[dict_idx].column_type_utf8.rfind("char", 0) == 0) {
      col_len = columns[i]["char_length"].GetInt();
    } else {
      std::cerr << "Unsupported column type: " 
                << dict_cols[dict_idx].column_type_utf8 << std::endl;
      return -1;
    }
    dict_cols[dict_idx].char_length = col_len;

    // Save the offset of this column in offsets_
    offsets_[dict_idx] = current_offset;

    // Advance the running offset
    current_offset += col_len;
    dict_idx++;
  }

  // 5) Append hidden columns: TRX_ID (6 bytes) and ROLL_PTR (7 bytes).
  //    (In older MySQL versions, TRX_ID might be 6 or 8 bytes, etc. 
  //     The code below follows the original approach: 6 for trx_id, 7 for roll_ptr.)
  offsets_[dict_idx] = current_offset; // TRX_ID
  current_offset += 6; 
  dict_idx++;

  offsets_[dict_idx] = current_offset; // ROLL_PTR
  current_offset += 7; 
  dict_idx++;

  // 6) offsets_[2] can store the total row length if the code needs it
  offsets_[2] = current_offset;

  // Return success
  return 0;
}
```

### What’s Changed?

- We **loop** through *all* user columns from the JSON.  
- We do **not** hardcode the “first column is offset 3, second column is offset 4, etc.”  
- We store each column’s offset into `offsets_[3 + i]`, and keep a `current_offset` that grows as we add columns.  
- After the user columns, we add the two hidden fields (`trx_id`, `roll_ptr`).

---

## 2. Rewrite `ShowRecord(...)` to Loop Over All Columns

Below is an updated `ShowRecord()` that displays **all user columns** plus the hidden columns at the end. If you do **not** want to display the hidden columns, you can omit the last part.

```cpp
void ShowRecord(rec_t *rec) {

  ...

  // Number of user columns
  const size_t n_user_cols = offsets_[1];

  // Print each user column
  for (size_t i = 0; i < n_user_cols; i++) {
    // The dict_col index is i+3 (we started storing user columns at offset 3)
    const size_t dict_idx = i + 3;
    // The offset within the record
    ulint col_offset = offsets_[dict_idx];

    // Column name
    printf("%s: ", dict_cols[dict_idx].col_name.c_str());

    // Print according to type
    if (dict_cols[dict_idx].column_type_utf8 == "int") {
      // In InnoDB, int fields in a clustered index often have the “sign bit” flipped
      // for correct sort order. The original code used XOR 0x80000000.
      // If that’s your preference, keep it:
      uint32_t val = mach_read_from_4(rec + col_offset) ^ 0x80000000;
      printf("%u\n", val);
    } else {
      // e.g. char(N)
      int len = dict_cols[dict_idx].char_length;
      printf("%.*s\n", len, rec + col_offset);
    }
  }

  // 2 hidden columns are stored after all user columns:
  //   offsets_[n_user_cols+3] => TRX_ID offset
  //   offsets_[n_user_cols+4] => ROLL_PTR offset
  ulint trx_id_offset  = offsets_[n_user_cols + 3];
  ulint roll_ptr_offset = offsets_[n_user_cols + 4];

  // The original code used 6 bytes for TRX_ID, 7 for ROLL_PTR.
  // You can read them with mach_read_from_6 / mach_read_from_7 or similar:
  uint64_t trx_id = mach_read_from_6(rec + trx_id_offset);
  // For roll_ptr, you might do mach_read_from_7(...) or treat it as 7 bytes, etc.
  // The simplest might be reading 8 bytes but ignoring 1? Or a dedicated function:
  // If your code does not define mach_read_from_7, you can create it or do a manual approach.
  // For brevity, assume we have one:
  uint64_t roll_ptr = mach_read_from_7(rec + roll_ptr_offset);

  printf("trx_id: %llu\n", (unsigned long long) trx_id);
  printf("roll_ptr: %llu\n", (unsigned long long) roll_ptr);
}
```

### Key Points

- We dynamically loop over **all** user columns (so composite PK is no problem).  
- Each INT column is sign-flipped via `^ 0x80000000`, as in your original code. (In reality, InnoDB does that for key columns, but you can do it for all `INT` columns if your table is a clustered index. Adapt to your use case if needed.)  
- After user columns, we read the hidden fields from offsets `[n_user_cols + 3]` and `[n_user_cols + 4]`.  

---

## 3. Why This Fixes Composite Keys

- With multiple primary-key columns, InnoDB stores them all in order in the clustered index record. The original code only accounted for exactly one user column at a fixed offset. Once a second PK column appeared, the offset calculation was **incorrect**.  
- Now, we push each user column into `offsets_[]` in a loop. Whether you have 1 or 10 PK columns, they’ll be handled in order.  
- The hidden columns are appended **after** all user columns.  

As long as your JSON “SDI” file has the columns in the same order they appear in the clustered index (which it usually does by default), the above approach will parse them correctly—including composite primary keys.

---

## 4. Additional Notes

- **Sign-Flipping Logic**: The `^ 0x80000000` trick is usually done for **all** integer columns in the index to maintain a correct lexicographic ordering in the B-tree. If you only want that for **primary-key** columns (and not for plain `INT` columns that aren’t part of the key), you’d need to mark which columns are part of the PK in your JSON and only flip those.  
- **6 vs 8 Bytes for `trx_id`**: In InnoDB 5.6+ / 5.7+, the **transaction ID** can be stored in 6 or 8 bytes (depending on version). The code above sticks to the older 6-byte approach (like your original snippet). If your table uses 8-byte transaction IDs, adapt accordingly.  
- **mach_read_from_7**: Not always part of older MySQL headers. If your environment lacks it, you can implement your own function or simply do a manual 7-byte read. Alternatively, read 8 bytes and mask out the top bits.  

Once you update those two functions (`rec_init_offsets()` and `ShowRecord()`), your program should **correctly parse** tables with **any** number of columns—including composite PKs.