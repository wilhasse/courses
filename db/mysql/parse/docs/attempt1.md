------

### 1. Attempt #1: Compile Under `innobase` (Percona Server)

1. **Location**: Created a new directory inside `storage/innobase` (e.g., `demo_decompress`) with a simple `CMakeLists.txt` and a `.cc` file that calls `page_zip_decompress()` or similar functions.

2. Linking Issues

   :

   - InnoDB in MySQL 8 (and Percona Server 8) has grown to rely on many server-wide headers and symbols. For example, references to `mysql_services`, `srv_page_size`, or replication code frequently appear.
   - This meant the “innobase” static library alone (`libinnobase.a`) wasn’t enough. The linker started demanding symbols from the rest of MySQL (e.g., `mysql_components_handle_std_exception` or MEB/hotbackup references).
   - Attempting to add more libraries (e.g., `sql_main`, `mysys`, `strings`, etc.) led to an avalanche of undefined references because Percona patches tie InnoDB to additional code paths (like `xtrabackup` or “MEB” hooks).

Result: **Could not cleanly produce a standalone binary** in the `innobase` folder without pulling nearly the entire MySQL/Percona codebase into the link.

------

### 2. Attempt #2: Leveraging Percona XtraBackup Code

1. **Why XtraBackup?**
   - XtraBackup does many of the same tasks needed here (reads `.ibd`, decrypts pages, deals with compression, loads keyring, etc.).
   - Thought it might be easier to piggyback on XtraBackup’s existing build system (which compiles InnoDB code in a “hotbackup-friendly” manner).
2. **Obstacles**:
   - XtraBackup maintains its own patches and references (e.g., `UNIV_HOTBACKUP`) to merge InnoDB code with XtraBackup code.
   - The “innobase” directory in XtraBackup can contain modifications that expect other xtrabackup sources to be present.
   - Building a simple .cc within XtraBackup triggered references to internal XtraBackup functions (like `xb_set_encryption(...)`) or variables (`use_dumped_tablespace_keys`) that do not exist in standard MySQL InnoDB.
   - Similarly, attempting to disable `UNIV_HOTBACKUP` caused missing headers (`hb_univ.i`, or `../meb/mutex.h`) or introduced other conditional references.

Result: **Still led to major link or compilation problems** because XtraBackup’s InnoDB is entangled with its own code hooks.

------

### 3. Attempt #3: Studying `innochecksum` in MySQL 8

1. **Why `innochecksum`?**
   - `innochecksum` is a minimal offline utility in MySQL 8 that reads InnoDB `.ibd` (and `.ibd`-like) pages, verifies checksums, and can decompress them.
   - It only links to a few libraries: `innodb_zipdecompress`, `mysys`, and so forth, *without* pulling in the entire SQL layer or replication code.
2. **Observations**:
   - `innochecksum` and another tool `ibd2sdi` rely on compile definitions like `UNIV_NO_ERR_MSGS`, `UNIV_LIBRARY`, and `DBUG_OFF` to avoid referencing big chunks of server logic.
   - They also link only the minimal needed libraries—so they are a blueprint for building smaller “standalone” InnoDB utilities.
   - However, these tools do not handle encryption or keyring integration. They also rely on the code in `innodb_zipdecompress` but do not show how to integrate the Percona encryption code.

------

### 4. `page0zip` vs. `zipdecompress`

- **`page0zip.cc`** in MySQL 8 historically handled compressed page logic. Over time, MySQL 8 introduced or refactored “page_zip_decompress_low” and related code into a smaller library (`innodb_zipdecompress`).
- The rationale was partly to allow external tools (like `innochecksum`) to link a small set of functions for decompression. However, *in practice,* you still encounter dependencies on InnoDB’s internal macros or server variables (like `srv_page_size`), so it is not entirely standalone.

------

### 5. Decrypting Pages: Studying XtraBackup Keyring Integration

1. Encryption Logic

   :

   - Found in `storage/innobase/os/os0enc.cc` and `os0enc.h`—these handle loading tablespace keys, decrypting pages, etc.
   - Percona Server also references a “keyring” plugin for retrieving encryption keys.

2. XtraBackup Keyring Setup

   :

   - In files like `keyring_components.cc`, `keyring_plugins.cc`, etc. XtraBackup will initialize the plugin, fetch the encryption key, and pass it to InnoDB logic that decrypts pages.

3. Implication

   :

   - If your custom tool needs to decrypt `.ibd` pages, you also have to replicate or invoke the XtraBackup (or MySQL server) logic for loading the keyring. That can bring in a large chunk of code or the actual plugin library (`.so`).

------

### 6. Conclusions & Next Steps

1. **Significant Ties to the Full Codebase**
   - InnoDB is *not* trivially standalone in MySQL 8 or Percona 8. Many references exist to server components, replication, or XtraBackup.
   - Stripping those references or mocking them out is a nontrivial job if you want a minimal offline utility.
2. **Potential Approaches**:
   - **(A)** Extend or clone an existing offline utility in Percona/MySQL, like `innochecksum` or `ibd2sdi`, which already compiles as a smaller scope. Then add the encryption calls from XtraBackup’s `os0enc.cc`, hooking the keyring logic.
   - **(B)** Build directly from the XtraBackup codebase, effectively writing a new “mini” xtrabackup-like utility that reuses their encryption and page parse code. But you’ll need to link in a lot of XtraBackup internals.
   - **(C)** If it’s purely for personal study or proof-of-concept, manually replicate the relevant encryption/decompression logic in your code, *inlined* or simplified, to avoid linking the entire server.
3. **Big Takeaway**:
   - MySQL 8’s InnoDB code is heavily integrated with other server modules.
   - XtraBackup’s InnoDB code is integrated with its own backup logic.
   - “Stand-alone” usage is possible in principle (like `innochecksum`), but bridging encryption or advanced features typically requires many additional symbols from the server or XtraBackup.

