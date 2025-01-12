#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "innodb_page.h"

#define MAX_TABLE_FIELDS 50

typedef struct {
    char* name;
    int type;
    bool can_be_null;
    int fixed_length;
    int min_length;
    int max_length;
} field_def_t;

typedef struct {
    char* name;
    unsigned fields_count;
    unsigned n_nullable;
    field_def_t fields[MAX_TABLE_FIELDS];
    ulint data_min_size;
    ulint data_max_size;
    ulint min_rec_header_len;
} table_def_t;

static bool debug = false;
static table_def_t g_table;
static bool g_has_table = false;
static unsigned long records_expected_total = 0;
static unsigned long records_dumped_total = 0;
static int records_lost = 0;

void dump_record_data(const byte *record, ulint length) {
    printf("Record data (%lu bytes): ", length);
    for (ulint i = 0; i < length; i++) {
        printf("%02x ", record[i]);
    }
    printf("\n");
}

bool check_page(const page_t *page, unsigned *n_records)
{
    bool comp = page_is_compact(page);
    ulint infimum = page_get_infimum_offset(comp);
    ulint supremum = page_get_supremum_offset(comp);
    const byte *page_ptr = (const byte *)page;

    if (debug) {
        printf("check_page(): comp=%d, inf=%lu, sup=%lu\n",
               (int)comp, infimum, supremum);
    }

    /* Start at infimum */
    ulint curr = infimum;
    int rec_count = 0;

    /* Walk the record chain */
    while (curr != supremum && rec_count < 1000) {  // Added safety limit
        if (curr < FIL_PAGE_DATA || curr >= UNIV_PAGE_SIZE - 8) {
            if (debug) printf("Invalid offset: %lu\n", curr);
            return false;
        }

        if (debug) {
            printf("  Record at offset %lu: ", curr);
            dump_record_data(page_ptr + curr, 20); // Dump first 20 bytes
        }

        ulint next = page_get_next_offset(page, curr, comp);
        
        if (next <= curr || next >= UNIV_PAGE_SIZE - 8) {
            if (debug) printf("Invalid next pointer: %lu\n", next);
            return false;
        }

        curr = next;
        rec_count++;
    }

    if (curr != supremum) {
        if (debug) printf("Chain did not reach supremum\n");
        return false;
    }

    *n_records = rec_count - 1; // Don't count infimum
    if (debug) printf("Found %d user records\n", *n_records);
    
    return true;
}

void process_ibpage(page_t *page, bool hex_output)
{
    ulint page_id = page_get_page_no(page);
    bool comp = page_is_compact(page);

    printf("-- process_ibpage() Page id: %lu, comp=%d\n",
           page_id, (int)comp);

    unsigned expected_records = 0;
    bool valid_chain = check_page(page, &expected_records);
    ulint n_recs_in_header = page_get_n_recs(page);

    printf("-- check_page() => is_valid_chain=%d, expected_records=%u, "
           "Page Header N_RECS=%lu\n",
           (int)valid_chain, expected_records, n_recs_in_header);

    if (valid_chain) {
        if (debug) {
            printf("Valid record chain found! Starting record processing...\n");
            /* Here we would process individual records */
        }
    }

    bool is_leaf = page_is_leaf(page);
    if (is_leaf && n_recs_in_header > 0) {
        records_expected_total += n_recs_in_header;
    }
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <tablename.ibd> [--debug]\n", argv[0]);
        return 1;
    }
    
    const char* ibd_path = argv[1];
    if (argc > 2 && strcmp(argv[2], "--debug") == 0) {
        debug = true;
    }

    int fd = open(ibd_path, O_RDONLY);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        perror("fstat");
        close(fd);
        return 1;
    }

    const size_t page_size = UNIV_PAGE_SIZE;
    size_t page_count = (size_t)(st.st_size / page_size);

    unsigned char *page_buf = malloc(page_size);
    if (!page_buf) {
        fprintf(stderr, "Failed to allocate page buffer\n");
        close(fd);
        return 1;
    }

    printf("Processing %zu pages...\n\n", page_count);

    for (size_t i = 0; i < page_count; i++) {
        off_t offset = (off_t)(i * page_size);
        ssize_t ret = pread(fd, page_buf, page_size, offset);
        if (ret < (ssize_t)page_size) {
            fprintf(stderr, "Short read or error at page %zu\n", i);
            break;
        }
        process_ibpage((page_t*)page_buf, false);
    }

    free(page_buf);
    close(fd);

    printf("\n===== Summary =====\n");
    printf("Records expected total: %lu\n", records_expected_total);
    printf("Records dumped total:   %lu\n", records_dumped_total);
    printf("Lost any records? %s\n", records_lost ? "YES" : "NO");

    return 0;
}
