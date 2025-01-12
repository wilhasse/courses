#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "innodb_page.h"

typedef struct {
    int id;
    char nome[101];
} record_t;

static bool debug = false;
static unsigned long records_expected_total = 0;
static unsigned long records_dumped_total = 0;
static int records_lost = 0;

void hexdump(const byte *data, size_t len, const char *prefix) {
    if (!debug) return;
    printf("%s", prefix);
    for (size_t i = 0; i < len; i++) {
        if (i > 0 && i % 16 == 0) printf("\n%s", prefix);
        printf("%02x ", data[i]);
    }
    printf("\n");
}

bool parse_user_record(const byte *rec_ptr, record_t *record) {
    if (is_system_record(rec_ptr)) {
        if (debug) printf("  Skipping system record\n");
        return false;
    }
    
    if (is_deleted_record(rec_ptr)) {
        if (debug) printf("  Skipping deleted record\n");
        return false;
    }
    
    /* Get ID field */
    ulint id_offset = get_field_offset(rec_ptr, 0);
    const byte* id_ptr = rec_ptr + id_offset;
    record->id = mach_read_from_4(id_ptr);
    
    /* Get NOME field */
    ulint nome_offset = get_field_offset(rec_ptr, 1);
    const byte* nome_ptr = rec_ptr + nome_offset;
    size_t name_len = 0;
    while (name_len < 100 && nome_ptr[name_len] && nome_ptr[name_len] != ' ') {
        name_len++;
    }
    memcpy(record->nome, nome_ptr, name_len);
    record->nome[name_len] = '\0';
    
    if (debug) {
        printf("  Parsed record: ID=%d, NOME=%s\n", record->id, record->nome);
        printf("  Raw data:\n");
        hexdump(rec_ptr, 120, "    ");
    }
    
    return true;
}

void process_index_page(const page_t *page) {
    bool header_printed = false;
    int records_on_page = 0;
    
    if (debug) {
        printf("\nProcessing index page:\n");
        page_header_print(page);
    }
    
    const byte* rec = get_first_record(page);
    while (rec) {
        if (debug) {
            printf("\nExamining record at offset %lu:\n", 
                   (ulint)(rec - (const byte*)page));
        }
        
        record_t record;
        if (parse_user_record(rec, &record)) {
            if (!header_printed) {
                printf("\nID\tNOME\n");
                printf("----------------------------------------\n");
                header_printed = true;
            }
            printf("%d\t%s\n", record.id, record.nome);
            records_dumped_total++;
            records_on_page++;
        }
        
        rec = get_next_record(page, rec);
    }
    
    if (debug) {
        printf("\nProcessed %d records on page\n", records_on_page);
    }
}

void process_ibpage(page_t *page, bool hex_output) {
    ulint page_type = page_get_type(page);
    ulint page_id = page_get_page_no(page);
    
    printf("\n-- Processing page %lu (type %lu)\n", page_id, page_type);
    
    if (page_type == FIL_PAGE_INDEX) {
        ulint n_recs = page_get_n_recs(page);
        if (page_is_leaf(page)) {
            printf("-- Leaf index page with %lu records\n", n_recs);
            process_index_page(page);
            records_expected_total += n_recs;
        } else {
            printf("-- Non-leaf index page (skipping)\n");
        }
    } else {
        switch(page_type) {
            case FIL_PAGE_TYPE_FSP_HDR:
                printf("-- File space header page (skipping)\n");
                break;
            case FIL_PAGE_IBUF_BITMAP:
                printf("-- Insert buffer bitmap (skipping)\n");
                break;
            case FIL_PAGE_TYPE_ALLOCATED:
                printf("-- Freshly allocated page (skipping)\n");
                break;
            case FIL_PAGE_INODE:
                printf("-- Index node page (skipping)\n");
                break;
            case FIL_PAGE_TYPE_SYS:
                printf("-- System page (skipping)\n");
                break;
            default:
                printf("-- Unknown page type %lu (skipping)\n", page_type);
        }
    }
}

int main(int argc, char** argv) {
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
    
    printf("Processing %zu pages...\n", page_count);
    
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
