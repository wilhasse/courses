#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "innodb_page.h"

/* Custom table definitions */
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
    field_def_t fields[50];
    ulint data_min_size;
    ulint data_max_size;
    ulint min_rec_header_len;
} table_def_t;

#define FT_INT  1
#define FT_CHAR 2
#define MAX_TABLE_FIELDS 50
#define record_extra_bytes 6

/* Global variables */
static bool debug = false;
static table_def_t g_table;
static bool g_has_table = false;
static unsigned long records_expected_total = 0;
static unsigned long records_dumped_total = 0;
static int records_lost = 0;
static bool deleted_records_only = false;
static bool undeleted_records_only = true;

/* Function declarations */
bool check_page(const page_t *page, unsigned *n_records);
void process_ibpage(page_t *page, bool hex_output);
bool init_table_definition(void);

/* Function implementations */
bool check_page(const page_t *page, unsigned *n_records)
{
    bool comp = page_is_compact(page);
    ulint infimum = (comp ? PAGE_NEW_INFIMUM : PAGE_OLD_INFIMUM);
    ulint supremum = (comp ? PAGE_NEW_SUPREMUM : PAGE_OLD_SUPREMUM);

    if (debug) {
        printf("check_page(): comp=%d, inf=%u, sup=%u\n",
               (int)comp, (unsigned)infimum, (unsigned)supremum);
    }

    ulint p_prev = 0;
    ulint p = infimum;
    *n_records = 0;

    const int max_recs = UNIV_PAGE_SIZE / 5;
    int rec_count = 0;

    while (1) {
        ulint next_p = page_offset_get_next(page, p, comp);

        if (p < 2 || p >= UNIV_PAGE_SIZE) {
            if (debug) printf("check_page(): pointer out of range: p=%u\n", (unsigned)p);
            return false;
        }
        if (next_p == p || next_p >= UNIV_PAGE_SIZE) {
            if (debug) printf("check_page(): next pointer out of range: next_p=%u\n", 
                            (unsigned)next_p);
            return false;
        }
        if (p == p_prev) {
            if (debug) printf("check_page(): loop detected p=%u\n", (unsigned)p);
            return false;
        }

        p_prev = p;
        rec_count++;

        if (p == supremum) break;

        p = next_p;
        if (rec_count > max_recs) {
            if (debug) printf("check_page(): rec_count > max_recs => corruption\n");
            return false;
        }
    }

    *n_records = (rec_count < 2) ? 0 : rec_count - 2;
    return true;
}

void process_ibpage(page_t *page, bool hex_output)
{
    ulint page_id = page_get_page_no(page);
    bool comp = page_is_compact(page);

    printf("-- process_ibpage() Page id: %lu, comp=%d\n",
           (unsigned long)page_id, (int)comp);

    unsigned expected_records = 0;
    bool valid_chain = check_page(page, &expected_records);
    ulint n_recs_in_header = page_get_n_recs(page);

    printf("-- check_page() => is_valid_chain=%d, expected_records=%u, "
           "Page Header N_RECS=%lu\n",
           (int)valid_chain, expected_records, (unsigned long)n_recs_in_header);

    if (!g_has_table) {
        printf("[Error] No table definition loaded.\n");
        return;
    }

    bool is_leaf_page = page_is_leaf(page);
    if (is_leaf_page) {
        records_expected_total += n_recs_in_header;
        /* For now, we'll just track the expected records */
    }
}

bool init_table_definition(void)
{
    g_table.name = strdup("TESTE");
    g_table.fields_count = 2;
    g_table.n_nullable = 0;
    
    g_table.fields[0].name = strdup("ID");
    g_table.fields[0].type = FT_INT;
    g_table.fields[0].can_be_null = false;
    g_table.fields[0].fixed_length = 4;
    g_table.fields[0].min_length = 4;
    g_table.fields[0].max_length = 4;
    
    g_table.fields[1].name = strdup("NOME");
    g_table.fields[1].type = FT_CHAR;
    g_table.fields[1].can_be_null = false;
    g_table.fields[1].fixed_length = 0;
    g_table.fields[1].min_length = 0;
    g_table.fields[1].max_length = 400;

    g_table.data_min_size = 8;
    g_table.data_max_size = 421;
    g_table.min_rec_header_len = 5;

    g_has_table = true;
    return true;
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

    if (!init_table_definition()) {
        fprintf(stderr, "Failed to initialize table definition\n");
        return 1;
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
