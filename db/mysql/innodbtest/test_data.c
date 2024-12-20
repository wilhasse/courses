#include <stdio.h>
#include <string.h>
#include "innodb.h"

int main() {
    ib_err_t    err;
    ib_trx_t    trx;
    ib_id_t     table_id;
    ib_crsr_t   crsr;

    // Initialize and start up InnoDB
    err = ib_init();
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_init() failed: %s\n", ib_strerror(err));
        return 1;
    }

    err = ib_startup(NULL);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_startup() failed: %s\n", ib_strerror(err));
        return 1;
    }

    // The database and table should already exist from the previous run
    // Try to get the table id
    err = ib_table_get_id("test/test_table", &table_id);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_table_get_id failed: %s\n", ib_strerror(err));
        return 1;
    }

    // Start a transaction to read data
    trx = ib_trx_begin(IB_TRX_REPEATABLE_READ);
    if (!trx) {
        fprintf(stderr, "ib_trx_begin() failed\n");
        return 1;
    }

    // Open the table
    err = ib_cursor_open_table_using_id(table_id, trx, &crsr);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_cursor_open_table_using_id failed: %s\n", ib_strerror(err));
        ib_trx_rollback(trx);
        return 1;
    }

    // Create a search tuple: we expect a single INT column as PK from previous example
    ib_tpl_t search_tpl = ib_clust_search_tuple_create(crsr);
    if (!search_tpl) {
        fprintf(stderr, "ib_clust_search_tuple_create failed\n");
        ib_cursor_close(crsr);
        ib_trx_rollback(trx);
        return 1;
    }

    // Suppose we previously inserted id=1234
    err = ib_tuple_write_u32(search_tpl, 0, 1234UL);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_tuple_write_u32 failed: %s\n", ib_strerror(err));
        ib_tuple_delete(search_tpl);
        ib_cursor_close(crsr);
        ib_trx_rollback(trx);
        return 1;
    }

    int result;
    err = ib_cursor_moveto(crsr, search_tpl, IB_CUR_GE, &result);
    if (err == DB_SUCCESS && result == 0 && ib_cursor_is_positioned(crsr)) {
        // Found the row
        ib_tpl_t read_tpl = ib_clust_read_tuple_create(crsr);
        if (!read_tpl) {
            fprintf(stderr, "ib_clust_read_tuple_create(read) failed\n");
        } else {
            err = ib_cursor_read_row(crsr, read_tpl);
            if (err == DB_SUCCESS) {
                // Single INT column
                ib_u32_t val;
                err = ib_tuple_read_u32(read_tpl, 0, &val);
                if (err == DB_SUCCESS) {
                    printf("Found row: id=%u\n", (unsigned)val);
                } else {
                    fprintf(stderr, "ib_tuple_read_u32 failed: %s\n", ib_strerror(err));
                }
            } else {
                fprintf(stderr, "ib_cursor_read_row failed: %s\n", ib_strerror(err));
            }
            ib_tuple_delete(read_tpl);
        }
    } else {
        printf("Row not found.\n");
    }

    ib_tuple_delete(search_tpl);

    // Commit the transaction and close
    err = ib_trx_commit(trx);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_trx_commit failed: %s\n", ib_strerror(err));
    }

    err = ib_cursor_close(crsr);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_cursor_close failed: %s\n", ib_strerror(err));
    }

    err = ib_shutdown(IB_SHUTDOWN_NORMAL);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_shutdown() failed: %s\n", ib_strerror(err));
    }

    return 0;
}
