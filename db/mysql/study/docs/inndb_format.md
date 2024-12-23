Below is an overview of how the InnoDB row (and file) formats evolved from the early Embedded InnoDB 1.0.6.6750 days (when Innobase Oy was still an independent company) through to modern MySQL 8. We will look at:

1. **Original row formats and “Antelope” file format**
2. **Introduction of new row formats and the “Barracuda” file format**
3. **Why these changes were introduced**
4. **Impact on internal data structures**
5. **Which MySQL versions support which formats**

Throughout, keep in mind that “InnoDB” has gone through many refactorings and acquisitions (Innobase -> Oracle, MySQL -> Sun -> Oracle), which influenced how and when certain features appeared in MySQL.

------

## 1. Early InnoDB & the Original “Antelope” File Format

### Redundant Row Format (Pre-MySQL 5.0.3)

- **Origins**: In the earliest InnoDB versions (including embedded versions from Innobase), there was effectively just one row format, which now is retroactively referred to as **Redundant**.

- Characteristics

  :

  - Stores a 6-byte header plus additional overhead for each column.
  - Variable-length columns require extra bytes to store offsets.
  - Overall overhead can be relatively high (especially for many columns).

- **File Format**: We now call that original file format **Antelope**, which supports only the Redundant and Compact row formats (the latter was added slightly later).

When MySQL integrated InnoDB, early MySQL 4.x and 5.0 (before 5.0.3) used Redundant by default.

### Compact Row Format (Introduced in MySQL 5.0.3)

- **Motivation**: Reduce storage overhead.

- Characteristics

  :

  - Unlike Redundant, which stores length bytes for every column, Compact stores offset information for variable-length columns in a more optimized way.
  - Helps reduce row storage overhead, especially for tables with many variable-length columns.
  - Became the default row format in MySQL 5.0.3 onward (still part of Antelope file format).

So with MySQL up to around 5.1, you effectively had these two row formats:

1. **Redundant** (legacy, earliest)
2. **Compact** (improved overhead, the new default from 5.0.3 onward)

Both of these were collectively part of the **Antelope** file format.

------

## 2. The “Barracuda” File Format: Compressed & Dynamic Row Formats

With MySQL 5.5 (technically introduced experimentally in 5.1 but became mainstream in 5.5), Oracle introduced a new file format named **Barracuda**. Barracuda brought two new row formats:

1. **Compressed**
2. **Dynamic**

> **Note**: Barracuda is fully backward compatible with Antelope for read operations, but older MySQL versions (and older InnoDB engines) cannot necessarily handle the new row formats.

### Dynamic Row Format

- **Motivation**: Handle large columns more efficiently and reduce page overhead.

- Characteristics

  :

  - Closely resembles Compact format but large (variable-length) columns can be stored off-page (i.e., in overflow pages) with only a 20-byte pointer in the main row.
  - More flexible than Compact, especially for long `TEXT` or `BLOB` columns.
  - Reduces page splits and can improve performance for large columns.

### Compressed Row Format

- **Motivation**: On-disk compression to save space.

- Characteristics

  :

  - A table (or partition) can be stored in a compressed form at the page level (e.g., 8K compressed to 4K).
  - Internally still quite similar to Dynamic in how columns are offloaded.
  - Reduces disk usage and can improve I/O performance if CPU overhead of compress/decompress is manageable.

### Barracuda vs. Antelope

- **Antelope** supports Redundant & Compact.
- **Barracuda** supports all four row formats (Redundant, Compact, Compressed, Dynamic) but is the only file format that allows Compressed and Dynamic.

As of MySQL 5.5, **Barracuda** became available (it did not immediately become the *default* in all distributions until later releases). In MySQL 5.7 and 8.0, Barracuda is effectively the standard recommended format.

------

## 3. Why These Changes Were Introduced

1. **Reduced Storage Overhead**:
   - Redundant -> Compact: The main driver was to eliminate unnecessary bytes for each column offset.
   - Compact -> Dynamic/Compressed: Store large columns off-page, or compressed, to reduce page usage and improve I/O.
2. **Performance & Scalability**:
   - With larger data sets, InnoDB needed more efficient ways of handling big columns (especially BLOB/TEXT).
   - Off-page storage in Dynamic reduces page splits and B-Tree fragmentation.
3. **Flexibility with New Features**:
   - As MySQL evolved (5.6, 5.7, 8.0), features like partitioning, large page sizes, online DDL, transparent table encryption, etc., all required (or were aided by) improvements in how rows/columns are stored.
4. **Compatibility & Future Extensions**:
   - Barracuda was designed to be forward-looking. It was easier to enhance new InnoDB features without being restricted by the older Redundant/Compact layouts.

------

## 4. Impact on Internal Data Structures

### Page Format

- **InnoDB Page**: A typical page is 16 KB by default (though MySQL 5.7+ supports different page sizes: 4K, 8K, 16K, etc.).
- **Off-Page Storage**: With Dynamic and Compressed, large columns aren’t fully in the main page; they store pointers, improving how space is utilized in the main page.
- **Compression**: For Compressed row format, the page is kept in both compressed and uncompressed forms in the buffer pool, involving additional logic to handle page splits, merges, and dictionary operations.

### Row Header & Column Offsets

- **Redundant**: Multiple bytes of offset overhead per column.
- **Compact**: Minimizes overhead by referencing the row end and calculating lengths backwards.
- **Dynamic/Compressed**: Column data might not be fully in the main page, so offsets may point to external pages.

### Data Dictionary & Metadata

- In **older** InnoDB, the metadata (dictionary information) was partially stored in the shared tablespace (`ibdata1`) and also in `.frm` files for MySQL.
- As of **MySQL 8.0**, the server uses a **unified data dictionary** (eliminating `.frm` files and storing the dictionary in transactional system tables). While this doesn’t directly change the *row format*, it’s part of the broader internal architectural changes that accompanied Barracuda usage as the de-facto standard.

------

## 5. Which MySQL Versions Support What?

Below is a simplified version timeline (not exhaustive, but covers major changes relevant to InnoDB row formats):

| MySQL Version               | Default Row Format & Key File Format Changes                 |
| --------------------------- | ------------------------------------------------------------ |
| **MySQL 4.x** (historical)  | InnoDB Redundant row format only.                            |
| **MySQL 5.0** (up to 5.0.2) | Mostly Redundant.                                            |
| **MySQL 5.0.3+**            | Compact row format introduced, becomes default. Still “Antelope”. |
| **MySQL 5.1**               | Antelope (Redundant, Compact) is default; Barracuda is optional (experimental in some builds). |
| **MySQL 5.5**               | Barracuda becomes fully supported (Compressed, Dynamic). Not always default, but recommended. |
| **MySQL 5.6**               | InnoDB enhancements: better online DDL, improved Barracuda usage, partial page writes, etc. |
| **MySQL 5.7**               | Barracuda is strongly recommended. Many new InnoDB features assume Barracuda. |
| **MySQL 8.0**               | Barracuda is effectively the standard. Data dictionary changed (no .frm files), and the default row format is typically **DYNAMIC**. |

> **Note**: You can still force older row formats if needed for backward compatibility, but MySQL 8.0 strongly nudges the use of DYNAMIC or COMPRESSED.

------

## Summary

- **Embedded InnoDB 1.0.6.6750** was from the era when only the original (later called Redundant) format existed. Eventually MySQL 5.0 introduced **Compact** (still under the Antelope umbrella).
- The **Barracuda** file format (MySQL 5.5+) introduced two additional row formats: **Compressed** and **Dynamic**.
- The changes were driven by a need for **better space utilization**, **performance**, and **large column handling**.
- Internally, these changes impacted how **B-Trees store row data and how large columns are offloaded** to overflow pages. Compressed format adds page-level compression logic.
- As of **MySQL 8.0**, Barracuda is the default file format, with **DYNAMIC** row format generally favored for new InnoDB tables.

Overall, these row-format transitions reflect InnoDB’s evolution from a standalone storage engine into the highly optimized and feature-rich default engine that it is today in MySQL 8.