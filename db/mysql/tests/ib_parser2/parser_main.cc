#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// Include InnoDB & “undrop” style headers
#include "page0page.h"
#include "rem0rec.h"
#include "mach0data.h"
#include "tables_dict.h"

// (Optional) RapidJSON if you still want to load from JSON
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <fstream>
#include <iostream>

/********************************************************************
 * Global or static variables 
 ********************************************************************/
static bool debug = false;   // set to ‘true’ for verbose output

// Example table definition array. In undrop code, there might be multiple.
static table_def_t g_table;
static bool        g_has_table = false;

// We also keep some global counters like in undrop code:
unsigned long records_expected_total = 0;
unsigned long records_dumped_total   = 0;
int           records_lost           = 0;

// If you want to skip deleted vs. undeleted:
bool deleted_records_only   = false;
bool undeleted_records_only = true; // e.g. default to only “active” rows
int record_extra_bytes = 0; 

/** ibrec_init_offsets_new() => fill offsets array for a COMPACT record. */
bool ibrec_init_offsets_new(const page_t* page,const rec_t* rec,
                            table_def_t* table,ulint* offsets)
{
  ulint status = rec_get_status((rec_t*)rec);
  if (status != REC_STATUS_ORDINARY) {
    return false;
  }
  // set #fields
  rec_offs_set_n_fields(offsets, (ulint)table->fields_count);

  const unsigned char* nulls = (const unsigned char*)rec - (REC_N_NEW_EXTRA_BYTES + 1);
  const unsigned char* lens  = nulls - ((table->n_nullable + 7) / 8);

  ulint offs = 0;
  ulint null_mask = 1;

  for (ulint i = 0; i < (ulint)table->fields_count; i++) {
    field_def_t* fld = &table->fields[i];
    bool is_null = false;

    if (fld->can_be_null) {
      if (null_mask == 0) {
        nulls--;
        null_mask = 1;
      }
      if ((*nulls & null_mask) != 0) {
        is_null = true;
      }
      null_mask <<= 1;
    }

    ulint len_val;
    if (is_null) {
      len_val = offs | REC_OFFS_SQL_NULL;
    } else {
      if (fld->fixed_length == 0) {
        ulint lenbyte = *lens--;
        if (fld->max_length > 255
            || fld->type == FT_BLOB
            || fld->type == FT_TEXT) {
          if (lenbyte & 0x80) {
            lenbyte <<= 8;
            lenbyte |= *lens--;
            offs += (lenbyte & 0x3fff);
            if (lenbyte & 0x4000) {
              len_val = offs | REC_OFFS_EXTERNAL;
              goto store_len;
            } else {
              len_val = offs;
              goto store_len;
            }
          }
        }
        offs += lenbyte;
        len_val = offs;
      } else {
        offs += (ulint)fld->fixed_length;
        len_val = offs;
      }
    }
store_len:
    offs &= 0xffff;
    ulint diff = (ulint)((const unsigned char*)rec + offs - (const unsigned char*)page);
    if (diff > (ulint)UNIV_PAGE_SIZE) {
      printf("Invalid offset => field %lu => %lu\n",
             (unsigned long)i, (unsigned long)offs);
      return false;
    }
    offsets[i+1] = len_val;
  }
  return true;
}

/********************************************************************
 * The “check_page()” function
 *   - Validates the record chain from INFIMUM -> SUPREMUM
 *   - Returns ‘true’ if we can parse it, ‘false’ if corrupted
 *   - Also returns the # of “expected” user records (minus infimum).
 ********************************************************************/
bool check_page(const page_t *page, unsigned *n_records)
{
    // is page compact format?
    bool comp = page_is_comp(page);

    // find infimum & supremum offsets
    ulint infimum  = (comp ? PAGE_NEW_INFIMUM  : PAGE_OLD_INFIMUM);
    ulint supremum = (comp ? PAGE_NEW_SUPREMUM : PAGE_OLD_SUPREMUM);

    if (debug) {
        printf("check_page(): comp=%d, inf=%u, sup=%u\n",
               (int)comp, (unsigned)infimum, (unsigned)supremum);
    }

    // Start scanning from infimum
    ulint p_prev = 0;
    ulint p      = infimum;
    *n_records   = 0;

    // We’ll allow at most some large number of records
    const int max_recs = UNIV_PAGE_SIZE / 5;
    int rec_count      = 0;

    while (true) {
        // The next pointer is either p + 2 bytes or the “redundant” style
        ulint next_p;
        if (comp) {
            // In compact, the “next record pointer” is stored at (p - 2)
            // Then we add it to p to get the next offset
            ulint offset_val = mach_read_from_2(page + p - 2);
            next_p = p + offset_val;
        } else {
            // Redundant
            next_p = mach_read_from_2(page + p - 2);
        }

        // Basic safety checks
        if (p < 2 || p >= UNIV_PAGE_SIZE) {
            // Corrupt
            if (debug) {
                printf("check_page(): pointer out of range: p=%u\n",
                       (unsigned)p);
            }
            return false;
        }
        if (next_p == p || next_p >= UNIV_PAGE_SIZE) {
            // corruption or end
            if (debug) {
                printf("check_page(): next pointer out of range: next_p=%u\n",
                       (unsigned)next_p);
            }
            return false;
        }
        if (p == p_prev) {
            // loop detected => corruption
            if (debug) {
                printf("check_page(): loop detected p=%u\n", (unsigned)p);
            }
            return false;
        }

        p_prev = p;
        rec_count++;

        if (p == supremum) {
            // we reached supremum => done
            break;
        }

        p = next_p;
        if (rec_count > max_recs) {
            if (debug) {
                printf("check_page(): rec_count > max_recs => corruption\n");
            }
            return false;
        }
    }

    // The “rec_count” includes Infimum and Supremum, so user records = rec_count - 2
    if (rec_count < 2) {
        // Means no user records
        *n_records = 0;
    } else {
        *n_records = rec_count - 2;
    }
    if (debug) {
        printf("check_page(): concluded OK, found %u user records.\n", 
               (unsigned)*n_records);
    }
    return true;
}

/********************************************************************
 * “check_for_a_record()” => 
 *   - calls ibrec_init_offsets_new() or old version if not comp
 *   - checks if DELETED flag is set or not
 *   - checks field sizes, etc.
 ********************************************************************/
bool check_for_a_record(page_t       *page,
                        rec_t        *rec,
                        table_def_t  *table,
                        ulint        *offsets)
{
    // Make sure offset is within page
    ulint offset_in_page = (ulint)((byte*)rec - (byte*)page);
    if (offset_in_page < table->min_rec_header_len + record_extra_bytes) {
        if (debug) {
            printf("check_for_a_record(): offset too small => %lu\n",
                   (unsigned long)offset_in_page);
        }
        return false;
    }

    // Check if record is deleted or ordinary
    int del_flag = rec_get_deleted_flag(rec, page_is_comp(page));
    if (deleted_records_only && del_flag == 0) {
        // we only want deleted
        return false;
    }
    if (undeleted_records_only && del_flag != 0) {
        // we only want “active” => skip
        return false;
    }

    // If compact => ibrec_init_offsets_new
    bool comp = page_is_comp(page);
    if (comp) {
        if (!ibrec_init_offsets_new(page, rec, table, offsets)) {
            return false;
        }
    }

    // Now check the total data size is within allowed bounds
    ulint data_sz = rec_offs_data_size(offsets);
    if (data_sz < table->data_min_size || data_sz > table->data_max_size) {
        if (debug) {
            printf("check_for_a_record(): data_size out of range => %lu\n", 
                   (unsigned long)data_sz);
        }
        return false;
    }

    // Also check individual field lengths if you want
    //if (!check_fields_sizes(rec, table, offsets)) {
    //    return false;
    //}

    // Optionally check domain constraints
    //if (!check_constraints(rec, table, offsets)) {
    //    return false;
    //}

    // If we got here => looks valid
    return true;
}

/********************************************************************
 * “process_ibrec()” => prints or handles the record.
 *   - In your minimal example, you can do a simple loop or a nice
 *     printing style. 
 *   - This matches the undrop approach: it prints each field, etc.
 ********************************************************************/
ulint process_ibrec(page_t *page,
                    rec_t  *rec,
                    table_def_t *table,
                    ulint  *offsets,
                    bool   hex_output)
{
    (void)page; // not used if just printing

    // Here’s a simple approach: 
    // Print all fields separated by ‘|’
    static bool printed_header = false;
    if (!printed_header) {
        // print column names
        for (unsigned i = 0; i < table->fields_count; i++) {
            printf("%s", table->fields[i].name);
            if (i < table->fields_count - 1) {
                printf("|");
            }
        }
        printf("\n");
        printed_header = true;
    }

    for (unsigned i = 0; i < table->fields_count; i++) {
        ulint len;
        const byte *ptr = rec_get_nth_field(rec, offsets, i, &len);
        if (len == UNIV_SQL_NULL) {
            // print "NULL"
            printf("NULL");
        } else {
            // naive: print as text
            // For int fields, you might parse them, etc.
            // For now, just do a “printf("%.*s")” style
            printf("%.*s", (int)len, (const char*)ptr);
        }
        if (i < table->fields_count - 1) {
            printf("|");
        }
    }
    printf("\n");

    // Return the “data_size” if you want. Otherwise 0
    return rec_offs_data_size(offsets);
}

/********************************************************************
 * “process_ibpage()” => 
 *   - calls check_page() to see if chain is valid
 *   - if valid, loops from Infimum to Supremum 
 *   - calls check_for_a_record() for each candidate
 *   - calls process_ibrec() if valid 
 ********************************************************************/
void process_ibpage(page_t *page, bool hex_output)
{
    // read page_id
    ulint page_id = mach_read_from_4(page + FIL_PAGE_OFFSET);
    bool  comp    = page_is_comp(page);

    printf("-- process_ibpage() Page id: %lu, comp=%d\n", 
           (unsigned long)page_id, (int)comp);

    // 1) check page
    unsigned expected_records = 0;
    bool valid_chain = check_page(page, &expected_records);

    // (In undrop code, we also look at PAGE_HEADER + PAGE_N_RECS)
    ulint n_recs_in_header = mach_read_from_2(page + PAGE_HEADER + PAGE_N_RECS);

    printf("-- check_page() => is_valid_chain=%d, expected_records=%u, "
           "Page Header N_RECS=%lu\n",
           (int)valid_chain, expected_records, (unsigned long)n_recs_in_header);

    // If we have no table definitions, bail
    if (!g_has_table) {
        printf("[Error] No table definition loaded.\n");
        return;
    }

    // 2) If chain is valid, get the starting offset => Infimum
    ulint infimum  = (comp ? PAGE_NEW_INFIMUM  : PAGE_OLD_INFIMUM);
    ulint supremum = (comp ? PAGE_NEW_SUPREMUM : PAGE_OLD_SUPREMUM);

    ulint offset;
    if (valid_chain) {
        // get next offset from infimum
        ulint b = mach_read_from_2(page + infimum - 2);
        offset  = (comp ? infimum + b : b);
    } else {
        // fallback if page is not valid
        offset = 100 + record_extra_bytes;
    }

    // 3) parse
    unsigned found_records = 0;
    while (true) {
        if (offset >= UNIV_PAGE_SIZE - record_extra_bytes) {
            break; 
        }
        if (valid_chain && offset == supremum) {
            // we are at supremum => done
            break;
        }

        rec_t *rec = (rec_t*)(page + offset);
        ulint offsets[MAX_TABLE_FIELDS + 2]; // local array for offsets

        // Try to parse for our single “g_table”
        if (check_for_a_record(page, rec, &g_table, offsets)) {
            // We found a valid record
            found_records++;
            process_ibrec(page, rec, &g_table, offsets, hex_output);

            // If the chain is valid, jump to the next offset
            //   using the same approach as in check_page() 
            ulint next_off;
            if (comp) {
                ulint val = mach_read_from_2(page + offset - 2);
                next_off = offset + val;
            } else {
                next_off = mach_read_from_2(page + offset - 2);
            }
            offset = next_off;
        } else {
            // Not a valid record => skip or move on
            if (valid_chain) {
                // If chain is valid, do the same “next record pointer”
                ulint next_off;
                if (comp) {
                    ulint val = mach_read_from_2(page + offset - 2);
                    next_off  = offset + val;
                } else {
                    next_off  = mach_read_from_2(page + offset - 2);
                }
                offset = next_off;
            } else {
                // If chain not valid, just increment offset by 1 
                offset++;
            }
        }
    }

    // Print some info
    printf("-- Page %lu => found_records=%u\n", 
           (unsigned long)page_id, found_records);

    // Optionally track “lost records”
    bool is_leaf_page = (mach_read_from_2(page + PAGE_HEADER + PAGE_LEVEL) == 0);
    if (is_leaf_page) {
        // accumulate stats
        records_expected_total += n_recs_in_header;
        records_dumped_total   += found_records;
        if (found_records != n_recs_in_header) {
            records_lost = 1;
        }
    }
}

/********************************************************************
 * main() or your parse-entry code:
 *   - open the .ibd file
 *   - discover page_count
 *   - read each page into a buffer
 *   - call process_ibpage(page_buf, ...)
 ********************************************************************/
int main(int argc, char** argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <tablename.ibd> <json_tabledef> [--debug]\n", argv[0]);
        return 1;
    }
    const char* ibd_path  = argv[1];
    const char* json_path = argv[2];
    if (argc > 3 && std::string(argv[3]) == "--debug") {
        debug = true;
    }

    // 1) Load table definition from JSON or your approach
    //    (Below is an example skeleton that sets up g_table)
    {
        // Suppose we do a simplified build_table_def_from_json() that
        // reads columns => fill g_table. 
        // For brevity, let’s skip the entire code. Just assume success:
        //   g_has_table = (build_table_def_from_json(&g_table, "TESTE") == 0);
        // In a real environment, do your real JSON loading:
        g_has_table = true;
        // Demo: set some min_rec_header_len, data_min_size, etc.
        g_table.name           = strdup("TESTE");
        g_table.fields_count   = 2;
        g_table.n_nullable     = 0;
        // define ID field
        g_table.fields[0].name         = strdup("ID");
        g_table.fields[0].type         = FT_INT;
        g_table.fields[0].can_be_null  = false;
        g_table.fields[0].fixed_length = 4;
        g_table.fields[0].min_length   = 4;
        g_table.fields[0].max_length   = 4;
        // define NOME field
        g_table.fields[1].name         = strdup("NOME");
        g_table.fields[1].type         = FT_CHAR;
        g_table.fields[1].can_be_null  = false;
        g_table.fields[1].fixed_length = 0; // variable
        g_table.fields[1].min_length   = 0;
        g_table.fields[1].max_length   = 400;

        g_table.data_min_size  = 8;   // a guess
        g_table.data_max_size  = 421; // from your debug
        g_table.min_rec_header_len = 5; // from your snippet
    }

    // 2) open the .ibd 
    int fd = open(ibd_path, O_RDONLY);
    if (fd < 0) {
        perror("open");
        return 1;
    }
    // get file size
    struct stat st;
    if (fstat(fd, &st) != 0) {
        perror("fstat");
        close(fd);
        return 1;
    }
    off_t filesize = st.st_size;
    const size_t kPageSize = UNIV_PAGE_SIZE; // 16k typically
    size_t page_count = (size_t)(filesize / kPageSize);

    // 3) read & parse each page
    std::vector<unsigned char> page_buf(kPageSize);
    for (size_t i = 0; i < page_count; i++) {
        off_t offset = (off_t)(i * kPageSize);
        ssize_t ret = pread(fd, page_buf.data(), kPageSize, offset);
        if (ret < (ssize_t)kPageSize) {
            fprintf(stderr, "Short read or error at page %zu\n", i);
            break;
        }
        // pass to process_ibpage
        process_ibpage(page_buf.data(), false /*hex_output*/);
    }

    close(fd);

    // 4) print final stats
    printf("\n===== Summary =====\n");
    printf("Records expected total: %lu\n", records_expected_total);
    printf("Records dumped total:   %lu\n", records_dumped_total);
    printf("Lost any records? %s\n", (records_lost ? "YES" : "NO"));

    return 0;
}
