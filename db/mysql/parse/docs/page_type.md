# Page types

Defined here storage/innobase/include/fil0fil.h

| Page Type | Value | Description |
|-----------|-------|-------------|
| FIL_PAGE_INDEX | 17855 | B-tree node, the most common page type used for storing table and index data |
| FIL_PAGE_RTREE | 17854 | R-tree node, used for spatial indexes |
| FIL_PAGE_UNDO_LOG | 2 | Stores undo information for transaction rollback |
| FIL_PAGE_TYPE_FSP_HDR | 8 | File space header, contains tablespace metadata |
| FIL_PAGE_TYPE_XDES | 9 | Extent descriptor, manages space allocation |
| FIL_PAGE_TYPE_BLOB | 10 | Uncompressed BLOB data storage |
| FIL_PAGE_COMPRESSED | 14 | Compressed page format |
| FIL_PAGE_ENCRYPTED | 15 | Encrypted page format B+Tree |
| FIL_PAGE_COMPRESSED_AND_ENCRYPTED | 16 | Both compressed and encrypted |
| FIL_PAGE_ENCRYPTED_RTREE | 17 | Encrypted page format R+Tree |
| FIL_PAGE_TYPE_LOB_DATA | 23 | Data pages for uncompressed LOBs |
| FIL_PAGE_TYPE_ZLOB_DATA | 26 | Data pages for compressed LOBs |

