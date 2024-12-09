# Project

**Innodb Space**  
https://github.com/baotiao/inno_space  

## Overview

This code is essentially a low-level InnoDB file inspection utility. It reads and interprets raw InnoDB data files (.ibd files) and other InnoDB-related files (like undo log files) directly from disk without using MySQL or InnoDB's actual runtime engine. By analyzing the on-disk format and using known InnoDB page and record layouts, the code prints out information about pages, indexes, undo logs, BLOB data, and more.

InnoDB data and index files use a specific on-disk page format. Each page is typically 16KB (default size) and is composed of several well-defined headers, trailers, and data sections. The code presented is essentially a command-line utility that:

1. Opens an InnoDB `.ibd` file or undo log file.
2. Reads specific pages (by page number).
3. Decodes and prints out metadata about those pages.
4. Shows detailed information such as:
   - File-level information like the space header.
   - Per-page `FIL` header info (like checksums, page number, next/prev page).
   - Page type determination (index page, undo page, blob page, etc.).
   - Parsing and listing records from index pages.
   - Showing BLOB page segments.
   - Displaying undo segment headers and rollback segment arrays.
   - Updating page checksums.
   - Removing pages from the linked page list (for demonstration).

The code references InnoDB internal structures and constants (e.g., `FIL_PAGE_TYPE`, `FIL_PAGE_OFFSET`, `FSP_HEADER_OFFSET`, etc.) to read raw bytes from the file and convert them to meaningful metadata.

## Key Concepts in InnoDB Page Format

A 16KB InnoDB page usually contains:

- **FIL Header (38 bytes)** at the beginning:
  - Checksum
  - Page number
  - Previous / Next page pointers
  - Log sequence number (LSN)
  - Page type (e.g., INDEX, UNDO, BLOB, etc.)

- **Identifying Page Types: InnoDB stores different types of pages. The code checks FIL_PAGE_TYPE to determine what the page holds:**

  - FIL_PAGE_INDEX: A B-Tree index page (data or non-leaf).
  - FIL_PAGE_UNDO_LOG: Undo log page.
  - FIL_PAGE_INODE: Inode page that tracks segments.
  - FIL_PAGE_TYPE_LOB_FIRST, FIL_PAGE_TYPE_LOB_INDEX, FIL_PAGE_TYPE_LOB_DATA: BLOB pages.
  - FIL_PAGE_TYPE_RSEG_ARRAY: Contains the Rollback Segment (rseg) array.
  - FIL_PAGE_TYPE_FSP_HDR: Tablespace header page.

Different page types have different internal layouts. For example:
- **FSP header page (page 0)**: Contains tablespace metadata such as space ID, size, lists of free extents, etc.
- **Index pages**: Contain an index header and a collection of records arranged in a B-tree structure.
- **LOB/BLOB pages**: Used to store overflow data (long columns) not fitting in the main record.
- **UNDO log pages**: Store undo entries for transactional rollbacks.
- **RSEG array page**: Contains rollback segment array entries.

- **Page-specific data** (like the FSP header for page 0, segment headers, index headers, data records, etc.)

- **FIL Trailer** (8 bytes) at the end:
  - Checksum repeated
  - Low 4 bytes of LSN

## Detailed Code Walkthrough

### Header Files and Includes

The code includes various headers from InnoDB internals (`fil0fil.h`, `page0page.h`, `fsp0fsp.h`, `fsp0types.h`, etc.) and from `rapidjson` for JSON parsing. It also includes standard C++ and POSIX headers for file I/O and memory operations.

Parsing Index Header (ShowIndexHeader):
For index pages, the code prints metadata stored in the page header:

- Directory slot count
- Garbage space
- Number of records
- Maximum transaction ID on the page
- Page level (0 for leaf pages, >0 for non-leaf)
- Index ID

Parsing Records: Parsing records from an InnoDB page is complex because InnoDB uses a compact row format. The code:

- Reads a JSON file (sdi_path) which describes the table schema (columns, data types, lengths).
- Uses internal offsets (offsets_ array) to figure out where each column is stored in the record.
- Processes special records like the "infimum" and "supremum" pseudo-records that mark the start and end of a page's record list.
- Decodes fields such as integers and character fields based on the schema info read from the JSON file.
- The ShowRecord() function uses these offsets to print each column's value from a raw record buffer.

BLOB and LOB Pages: InnoDB can store large column values ("Longtext", "Blob", etc.) outside of the main index page in separate pages. The code can:

- Identify LOB/BLOB pages (FIL_PAGE_TYPE_BLOB, LOB_FIRST, LOB_INDEX, LOB_DATA).
- Print out their headers and linked information.
- Functions like ShowBlobHeader(), ShowBlobFirstPage(), ShowBlobIndexPage(), and ShowBlobDataPage() print details from these page types.

Undo Log Pages: When reading undo log files, it uses ShowUndoFile() and ShowUndoPageHeader() to:

- Identify rollback segments
- Print undo page headers
- Show undo records (which contain previous versions of rows)

### Command-Line Interface

The `main()` function supports various flags:
- `-f <file>`: Specifies the `.ibd` or undo file path.
- `-p <page_num>`: Shows details of a specific page.
- `-d <page_num>`: Deletes a specific page (by removing it from the linked list).
- `-u <page_num>`: Updates the checksum of a specific page.
- `-c <command>`: A command that can be `list-page-type`, `index-summary`, `show-undo-file`, `show-records`, etc.

The `-s <sdi_path>` specifies a JSON (SDI) file that describes the table's schema, allowing the code to interpret record formats and column data.

### Reading Pages and FIL Header

**Function: `ShowFILHeader(uint32_t page_num, uint16_t* type)`**

- Seeks to `page_num * kPageSize` in the file.
- Reads the first 38 bytes of the page to decode:
  - Checksum
  - Page number
  - Previous/Next page links
  - LSN
  - Page type stored in the FIL header
- Prints out these details.
- `*type` returns the InnoDB page type (e.g., `FIL_PAGE_INDEX`, `FIL_PAGE_UNDO_LOG`, etc.).

### Showing Index Headers and Records

**Function: `ShowIndexHeader(uint32_t page_num, bool is_show_records)`**

- After reading the page, it prints InnoDB index page header fields:
  - Number of directory slots
  - Garbage space
  - Number of records
  - Max transaction ID on page
  - B-tree level of the page
  - Index ID
- It also checks for a "symbol table" used in newer InnoDB versions that store additional info.
- If `is_show_records` is true and the page type is an index page, it calls `rec_init_offsets()` and attempts to print records.
  
#### Record Offsets and Schema Interpretation

**Function: `rec_init_offsets()`**

- Reads a JSON file specified by `sdi_path` that describes table schema.
- From the schema, it determines column offsets within a record:
  - The code handles integer and `char` columns.
  - Stores these offsets in `offsets_` and column metadata in `dict_cols`.
  
**Function: `ShowRecord(rec_t *rec)`**

- Given a record pointer `rec` on an index page, this function:
  - Reads heap number, flags, and whether the record is deleted/minimum.
  - Prints column values by using `offsets_` and `dict_cols` to interpret the data type.
  - Converts integer fields with a known InnoDB offset (InnoDB stores INTEGER in a big-endian format with a bias).
  - For `char` columns, it prints strings by their length.

### Undo Pages and Rollback Segments

**Function: `ShowUndoPageHeader(uint32_t page_num)`**

- For undo log pages, it prints out:
  - Undo page type (insert or update)
  - Offsets for log records
  - Previous/Next undo page in the chain

**Function: `ShowRsegArray(uint32_t page_num, uint32_t* rseg_array = nullptr)`**

- Rollback segment (RSEG) arrays are stored on a specific page (often page 2 in older InnoDB formats).
- This function lists all RSEG pages from that array.

**Function: `ShowUndoRseg(uint32_t rseg_id, uint32_t page_num)`**

- Given an RSEG page, prints rollback segment header info, the history list, and tries to show the last undo log header.

### Space (Tablespace) Header and Extents

**Function: `ShowSpaceHeader()`**

- For page 0, prints the space header:
  - Space ID
  - Highest page number allocated
  - Free limit (the first free page in the file)
  - Next segment ID to be allocated

**Function: `ShowExtent()`**

- Shows the state of each extent. InnoDB groups pages into extents (64 consecutive pages).
- Each extent can be `FREE`, `ALLOCATED`, `FULL`, etc.
  
### Listing Page Types Across the Tablespace

**Function: `ShowSpacePageType()`**

- Iterates over all pages in the file.
- Groups consecutive pages of the same type together.
- Prints ranges of pages and their type.
  
### Index Summary and Free Space

**Function: `ShowIndexSummary()`**

- Scans the file and tries to locate root index pages of primary and secondary indexes.
- For each root page found, it retrieves the file segment inode of leaf and non-leaf segments.
- Prints out how many extents are free, partially free, and how many pages are reserved/used.
- Provides a summary of how much space could be reclaimed if you run `OPTIMIZE TABLE`.

### Dumping All Records

**Function: `DumpAllRecords()`**

- Attempts to traverse down from the root page of the primary index to the leaf pages.
- Once on leaf pages, it iterates through all leaf pages using the `next` pointer in the page header.
- Prints all user records after interpreting them via `ShowRecord()`.

### Modifying the Pages

**DeletePage(page_num)**

- Demonstrates how to remove a page from the doubly-linked list of pages.
- It finds the previous and next pages, updates their pointers, and re-calculates checksums.

**UpdateCheckSum(page_num)**

- Reads the page, calculates a new checksum using `buf_calc_page_crc32()`, and writes it back.

## Conclusion

This code provides an inside look into how InnoDB pages are structured and interconnected. By reading raw bytes from `.ibd` files, interpreting them according to InnoDB's internal page definitions, and printing human-readable information, it serves as a valuable tool for learning or debugging InnoDB storage internals.

While the code relies on many InnoDB-specific constants and macros (like `FIL_PAGE_PREV`, `FIL_PAGE_NEXT`, `PAGE_HEADER`, `FSP_HEADER_SIZE`, etc.), the general approach is:

1. **Seek to the page offset.**
2. **Read raw data into a buffer.**
3. **Use known offsets and macros to decode fields.**
4. **Interpret the fields based on InnoDB's on-disk format.**
5. **Print or manipulate the data accordingly.**

