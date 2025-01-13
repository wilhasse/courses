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
#include "tables_json.h"

// (Optional) RapidJSON if you still want to load from JSON
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <fstream>
#include <iostream>

/********************************************************************
 * Global or static variables
 ********************************************************************/
bool debug = true; // set to ‘true’ for verbose output
table_def_t table_definitions[1];

// Example table definition array. In undrop code, there might be multiple.
static table_def_t g_table;
static bool g_has_table = false;

// We also keep some global counters like in undrop code:
unsigned long records_expected_total = 0;
unsigned long records_dumped_total = 0;
int records_lost = 0;

// If you want to skip deleted vs. undeleted:
bool deleted_records_only = false;
bool undeleted_records_only = true; // e.g. default to only “active” rows

/** ibrec_init_offsets_new() => fill offsets array for a COMPACT record. */
inline ibool ibrec_init_offsets_new(page_t *page, rec_t* rec, table_def_t* table, ulint* offsets) {
	ulint i = 0;
	ulint offs;
	const byte* nulls;
	const byte* lens;
	ulint null_mask;
	ulint status = rec_get_status(rec);

	// Skip non-ordinary records
	if (status != REC_STATUS_ORDINARY) return FALSE;

	// First field is 0 bytes from origin point
	rec_offs_base(offsets)[0] = 0;

	// Init first bytes
	rec_offs_set_n_fields(offsets, table->fields_count);

	nulls = rec - (REC_N_NEW_EXTRA_BYTES + 1);
	lens = nulls - (table->n_nullable + 7) / 8;
	offs = 0;
	null_mask = 1;

	/* read the lengths of fields 0..n */
	do {
		ulint	len;
		field_def_t *field = &(table->fields[i]);

		/* nullable field => read the null flag */
		if (field->can_be_null) {
//			if (debug) printf("nullable field => read the null flag\n");
			if (!(byte)null_mask) {
				nulls--;
				null_mask = 1;
			}

			if (*nulls & null_mask) {
				null_mask <<= 1;
				/* No length is stored for NULL fields.
				We do not advance offs, and we set
				the length to zero and enable the
				SQL NULL flag in offsets[]. */
				len = offs | REC_OFFS_SQL_NULL;
				goto resolved;
			}
			null_mask <<= 1;
		}

		if (!field->fixed_length) {
//			if (debug) printf("Variable-length field: read the length\n");
			/* Variable-length field: read the length */
			len = *lens--;

			if (field->max_length > 255 || field->type == FT_BLOB || field->type == FT_TEXT) {
				if (len & 0x80) {
					/* 1exxxxxxx xxxxxxxx */
					len <<= 8;
					len |= *lens--;

					offs += len & 0x3fff;
					if (len	& 0x4000) {
						len = offs | REC_OFFS_EXTERNAL;
					} else {
						len = offs;
					}

					goto resolved;
				}
			}

			len = offs += len;
		} else {
			len = offs += field->fixed_length;
		}
	resolved:
        offs &= 0xffff;
		if (rec + offs - page > UNIV_PAGE_SIZE) {
			if (debug) printf("Invalid offset for field %lu: %lu\n", i, offs);
			return FALSE;
		}
		rec_offs_base(offsets)[i + 1] = len;
	} while (++i < table->fields_count);

	return TRUE;
}

inline ibool check_fields_sizes(rec_t *rec, table_def_t *table, ulint *offsets)
{
  int i;

  if (debug)
  {
    printf("\nChecking field lengths for a row (%s): ", table->name);
    printf("OFFSETS: ");
    unsigned long int prev_offset = 0;
    unsigned long int curr_offset = 0;
    for (i = 0; i < rec_offs_n_fields(offsets); i++)
    {
      curr_offset = rec_offs_base(offsets)[i];
      printf("%lu (+%lu); ", curr_offset, curr_offset - prev_offset);
      prev_offset = curr_offset;
    }
  }

  // check every field
  for (i = 0; i < table->fields_count; i++)
  {
    // Get field size
    ulint len = rec_offs_nth_size(offsets, i);
    if (debug)
      printf("\n - field %s(%lu):", table->fields[i].name, len);

    // If field is null
    if (len == UNIV_SQL_NULL)
    {
      // Check if it can be null and jump to a next field if it is OK
      if (table->fields[i].can_be_null)
        continue;
      // Invalid record where non-nullable field is NULL
      if (debug)
        printf("Can't be NULL or zero-length!\n");
      return FALSE;
    }

    // Check size of fixed-length field
    if (table->fields[i].fixed_length)
    {
      // Check if size is the same and jump to the next field if it is OK
      if (len == table->fields[i].fixed_length || (len == 0 && table->fields[i].can_be_null))
        continue;
      // Invalid fixed length field
      if (debug)
        printf("Invalid fixed length field size: %lu, but should be %u!\n", len, table->fields[i].fixed_length);
      return FALSE;
    }

    // Check if has externally stored data
    if (rec_offs_nth_extern(offsets, i))
    {
      if (debug)
        printf("\nEXTERNALLY STORED VALUE FOUND in field %i\n", i);
      if (table->fields[i].type == FT_TEXT || table->fields[i].type == FT_BLOB)
        continue;
      if (debug)
        printf("Invalid external data flag!\n");
      return FALSE;
    }

    // Check size limits for varlen fields
    if (len < table->fields[i].min_length || len > table->fields[i].max_length)
    {
      if (debug)
        printf("Length limits check failed (%lu < %u || %lu > %u)!\n", len, table->fields[i].min_length, len, table->fields[i].max_length);
      return FALSE;
    }

    if (debug)
      printf("OK!");
  }

  if (debug)
    printf("\n");
  return TRUE;
}

int check_page(page_t *page, unsigned int *n_records)
{
  int comp = page_is_comp(page);
  int16_t i, s, p, b, p_prev;
  int recs = 0;
  int max_recs = UNIV_PAGE_SIZE / 5;
  *n_records = 0;
  i = (comp) ? PAGE_NEW_INFIMUM : PAGE_OLD_INFIMUM;
  s = (comp) ? PAGE_NEW_SUPREMUM : PAGE_OLD_SUPREMUM;

  if (deleted_records_only == 1)
  {
    if (debug)
      printf("We look for deleted records only. Consider all pages are not valid\n");
    return 0;
  }
  if (debug)
    printf("Checking a page\nInfimum offset: 0x%X\nSupremum offset: 0x%X\n", i, s);
  p_prev = 0;
  p = i;
  while (p != s)
  {
    if (recs > max_recs)
    {
      *n_records = 0;
      if (debug)
        printf("Page is bad\n");
      return 0;
    }
    // If a pointer to the next record is negative - the page is bad
    if (p < 2)
    {
      *n_records = 0;
      if (debug)
        printf("Page is bad\n");
      return 0;
    }
    // If the pointer is bigger than UNIV_PAGE_SIZE, the page is corrupted
    if (p > UNIV_PAGE_SIZE)
    {
      *n_records = 0;
      if (debug)
        printf("Page is bad\n");
      return 0;
    }
    //  If we've already was here, the page is bad
    if (p == p_prev)
    {
      *n_records = 0;
      if (debug)
        printf("Page is bad\n");
      return 0;
    }
    p_prev = p;
    // Get next pointer
    if (comp)
    {
      b = mach_read_from_2(page + p - 2);
      p = p + b;
    }
    else
    {
      p = mach_read_from_2(page + p - 2);
    }
    if (debug)
      printf("Next record at offset: 0x%X (%d) \n", 0x0000FFFF & p, p);
    recs++;
  }
  *n_records = recs - 1; // - infinum record
  if (debug)
    printf("Page is good\n");
  return 1;
}

inline ibool check_for_a_record(page_t *page, rec_t *rec, table_def_t *table, ulint *offsets)
{
  ulint offset, data_size;
  int flag;

  // Check if given origin is valid
  offset = rec - page;
  if (offset < record_extra_bytes + table->min_rec_header_len)
    return FALSE;
  if (debug)
    printf("ORIGIN=OK ");

  flag = rec_get_deleted_flag(rec, page_is_comp(page));
  if (debug)
    printf("DELETED=0x%X ", flag);
  // Skip non-deleted records
  if (deleted_records_only && flag == 0)
    return FALSE;

  // Skip deleted records
  if (undeleted_records_only && flag != 0)
    return FALSE;

  // Get field offsets for current table
  int comp = page_is_comp(page);
  if (comp && !ibrec_init_offsets_new(page, rec, table, offsets))
    return FALSE;
  if (debug)
    printf("OFFSETS=OK ");

  // Check the record's data size
  data_size = rec_offs_data_size(offsets);
  if (data_size > table->data_max_size)
  {
    if (debug)
      printf("DATA_SIZE=FAIL(%lu > %ld) ", (long int)data_size, (long int)table->data_max_size);
    return FALSE;
   }
  if (data_size < table->data_min_size)
  {
    if (debug)
      printf("DATA_SIZE=FAIL(%lu < %lu) ", (long int)data_size, (long int)table->data_min_size);
    return FALSE;
   }
  if (debug)
    printf("DATA_SIZE=OK ");

  // Check fields sizes
  //if (!check_fields_sizes(rec, table, offsets))
  //  return FALSE;
  if (debug)
    printf("FIELD_SIZES=OK ");

  // This record could be valid and useful for us
  return TRUE;
}

/********************************************************************
 * “process_ibrec()” => prints or handles the record.
 *   - In your minimal example, you can do a simple loop or a nice
 *     printing style.
 *   - This matches the undrop approach: it prints each field, etc.
 ********************************************************************/
ulint process_ibrec_raw(
    page_t       *page,
    rec_t        *rec,
    table_def_t  *table,
    ulint        *offsets,
    bool          hex_output)
{
    // Let's get the total data length for *all* fields
    ulint data_size = rec_offs_data_size(offsets);

    // If you like, you can also see how many fields the parser sees:
    ulint num_fields = rec_offs_n_fields(offsets);

    // Or simply rec_offs_n_fields(offsets). In COMPACT, hidden columns can be included if you have them in table->fields_count.

    printf("--- Row at rec=%p, data_size=%lu bytes, num_fields=%lu\n",
           rec, (unsigned long)data_size, (unsigned long)num_fields);

    // Dump the row header bytes (the “extra” overhead). Typically 5 bytes in COMPACT
    // but let's just do a small chunk if you want:
    const byte *rec_start = (const byte*) rec;
    // Let's say we dump 10 bytes of the record header just for debugging:
    printf("Header bytes (up to 10): ");
    for (int i = 0; i < 10; i++) {
        printf("%02X ", rec_start[i]);
    }
    printf("\n");

    // Now dump the *entire* record user data portion:
    // Because rec_offs_data_size(offsets) is the total length from the record start
    // to the last field, ignoring the record header. So let's figure out
    // where that data region begins, typically rec_get_nth_field(rec, offsets, 0, &len) or so.

    // If you just want the entire record from rec_start up to data_size + header, do:
    // BUT careful, because InnoDB record has some new/old style offsets. We'll do a simpler approach:

    // We'll just iterate over each field in offsets and dump them raw:
    ulint n_fields = rec_offs_n_fields(offsets);
    for (ulint i = 0; i < n_fields; i++)
    {
        ulint len;
        const byte *field_ptr = rec_get_nth_field(rec, offsets, i, &len);

        // If len == UNIV_SQL_NULL, skip or print a marker
        if (len == UNIV_SQL_NULL) {
            printf("Field #%lu is NULL\n", (unsigned long)i);
            continue;
        }

        // Otherwise, dump the field’s raw bytes:
        printf("Field #%lu, length=%lu:\n", (unsigned long)i, (unsigned long)len);
        if (hex_output) {
            // hex printing
            for (ulint b = 0; b < len; b++) {
                printf("%02X ", (unsigned)field_ptr[b]);
            }
            printf("\n");
        } else {
            // naive text printing (may contain garbage if the field is not ASCII)
            printf("\"");
            for (ulint b = 0; b < len; b++) {
                // For readability, replace unprintable bytes with dot
                unsigned char c = field_ptr[b];
                if (c >= 32 && c < 127) {
                    printf("%c", c);
                } else {
                    printf(".");
                }
            }
            printf("\"\n");
        }
    }

    // Return data_size so the caller can jump to next offset if desired
    return data_size;
}

inline ibool check_constraints(rec_t *rec, table_def_t *table, ulint *offsets)
{
  int i;
  ulint len_sum = 0;

  if (debug)
  {
    printf("\nChecking constraints for a row (%s) at %p:", table->name, rec);
    ut_print_buf(stdout, rec, 100);
  }

  // Check every field
  for (i = 0; i < table->fields_count; i++)
  {
    // Get field value pointer and field length
    ulint len;
    byte *field = rec_get_nth_field(rec, offsets, i, &len);
    if (debug)
      printf("\n - field %s(addr = %p, len = %lu):", table->fields[i].name, field, len);

    if (len != UNIV_SQL_NULL)
    {
      len_sum += len;
    }
    else
    {
      if (!rec_offs_comp(offsets))
      {
        len_sum += rec_get_nth_field_size(rec, i);
      }
    }

    // Skip null fields from type checks and fail if null is not allowed by data limits
    if (len == UNIV_SQL_NULL)
    {
      if (table->fields[i].has_limits && !table->fields[i].limits.can_be_null)
      {
        if (debug)
          printf("data can't be NULL");
        return FALSE;
      }
      continue;
    }
  }

  if (debug)
    printf("\nRow looks OK!\n");
  return TRUE;
}

void process_ibpage(page_t *page, bool hex)
{
  ulint page_id;
  rec_t *origin;
  ulint offsets[MAX_TABLE_FIELDS + 2];
  ulint offset, i;

  int is_page_valid = 0;
  int comp;
  unsigned int expected_records = 0;
  unsigned int expected_records_inheader = 0;
  unsigned int actual_records = 0;
  int16_t b, infimum, supremum;

  // Read page id
  page_id = mach_read_from_4(page + FIL_PAGE_OFFSET);
  if (debug)
    printf("Page id: %lu\n", page_id);

  if (table_definitions_cnt == 0)
  {
    fprintf(stderr, "There are no table definitions. Please check  include/table_defs.h\n");
    exit(EXIT_FAILURE);
  }
  is_page_valid = check_page(page, &expected_records);

  // comp == 1 if page in COMPACT format and 0 if REDUNDANT
  comp = page_is_comp(page);
  infimum = (comp) ? PAGE_NEW_INFIMUM : PAGE_OLD_INFIMUM;
  supremum = (comp) ? PAGE_NEW_SUPREMUM : PAGE_OLD_SUPREMUM;
  // Find possible data area start point (at least 5 bytes of utility data)
  if (is_page_valid)
  {
    b = mach_read_from_2(page + infimum - 2);
    offset = (comp) ? infimum + b : b;
  }
  else
  {
    offset = 100 + record_extra_bytes;
  }
  printf(", Records list: %s", is_page_valid ? "Valid" : "Invalid");
  expected_records_inheader = mach_read_from_2(page + PAGE_HEADER + PAGE_N_RECS);
  printf(", Expected records: (%u %u)", expected_records, expected_records_inheader);
  printf("\n");
  if (debug)
    printf("Starting offset: %lu (%lX). Checking %d table definitions.\n", offset, offset, table_definitions_cnt);

  // Walk through all possible positions to the end of page
  // (start of directory - extra bytes of the last rec)
  // is_page_valid = 0;
  while (offset < UNIV_PAGE_SIZE - record_extra_bytes && ((offset != supremum) || !is_page_valid))
  {
    // Get record pointer
    origin = page + offset;
    if (debug)
      printf("\nChecking offset: 0x%lX: ", offset);

    // Check all tables
    for (i = 0; i < table_definitions_cnt; i++)
    {
      // Get table info
      table_def_t *table = &(table_definitions[i]);
      if (debug)
        printf(" (%s) ", table->name);

      // Check if origin points to a valid record
      if (check_for_a_record(page, origin, table, offsets) && check_constraints(origin, table, offsets))
      {
        actual_records++;
        if (debug)
          printf("\n---------------------------------------------------\n"
                 "PAGE%lu: Found a table %s record: %p (offset = %lu)\n",
                 page_id, table->name, origin, offset);
        if (is_page_valid)
        {
          process_ibrec_raw(page, origin, table, offsets, hex);
          b = mach_read_from_2(page + offset - 2);
          offset = (comp) ? offset + b : b;
        }
        else
        {
          offset += process_ibrec_raw(page, origin, table, offsets, hex);
        }
        if (debug)
          printf("Next offset: 0x%lX", offset);
        break;
      }
      else
      {
        if (is_page_valid)
        {
          b = mach_read_from_2(page + offset - 2);
          offset = (comp) ? offset + b : b;
        }
        else
        {
          offset++;
        }
        if (debug)
          printf("\nNext offset: %lX", offset);
      }
    }
  }
  int leaf_page = mach_read_from_2(page + PAGE_HEADER + PAGE_LEVEL) == 0;
  int lost_records = (actual_records != expected_records) && (actual_records != expected_records_inheader);
  printf("-- Page id: %lu", page_id);
  printf(", Found records: %u", actual_records);
  printf(", Lost records: %s", lost_records ? "YES" : "NO");
  printf(", Leaf page: %s", leaf_page ? "YES" : "NO");
  printf("\n");
  if (leaf_page)
  {
    records_expected_total += expected_records_inheader;
    records_dumped_total += actual_records;
    if (lost_records)
    {
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
int main(int argc, char **argv)
{
  if (argc < 3)
  {
    fprintf(stderr, "Usage: %s <tablename.ibd> <json_tabledef> [--debug]\n", argv[0]);
    return 1;
  }
  const char *ibd_path = argv[1];
  const char *json_path = argv[2];
  if (argc > 3 && std::string(argv[3]) == "--debug")
  {
    debug = true;
  }

  // Load table definition from JSON or your approach
  load_ib2sdi_table_columns(argv[2]);

  // Build a table_def_t from g_columns
  static table_def_t g_table;
  if (build_table_def_from_json(&g_table, "TESTE") != 0) {
    std::cerr << "Failed to build table_def_t from JSON.\n";
    return 1;
  }

  // Add table
  table_definitions[0] = g_table;
  table_definitions_cnt = 1;

  // 2) open the .ibd
  int fd = open(ibd_path, O_RDONLY);
  if (fd < 0)
  {
    perror("open");
    return 1;
  }
  // get file size
  struct stat st;
  if (fstat(fd, &st) != 0)
  {
    perror("fstat");
    close(fd);
    return 1;
  }
  off_t filesize = st.st_size;
  const size_t kPageSize = UNIV_PAGE_SIZE; // 16k typically
  size_t page_count = (size_t)(filesize / kPageSize);

  // 3) read & parse each page
  std::vector<unsigned char> page_buf(kPageSize);
  for (size_t i = 0; i < page_count; i++)
  {
    off_t offset = (off_t)(i * kPageSize);
    ssize_t ret = pread(fd, page_buf.data(), kPageSize, offset);
    if (ret < (ssize_t)kPageSize)
    {
      fprintf(stderr, "Short read or error at page %zu\n", i);
      break;
    }
    // pass to process_ibpage
    if (i == 4)
    {
      init_table_defs(page_is_comp(page_buf.data()));
      process_ibpage(page_buf.data(), false /*hex_output*/);
    }
  }

  close(fd);

  // 4) print final stats
  printf("\n===== Summary =====\n");
  printf("Records expected total: %lu\n", records_expected_total);
  printf("Records dumped total:   %lu\n", records_dumped_total);
  printf("Lost any records? %s\n", (records_lost ? "YES" : "NO"));

  return 0;
}
