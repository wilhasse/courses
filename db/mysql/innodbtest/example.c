#include <stdio.h>
#include <string.h>
#include "innodb.h"

int main() {
    ib_err_t    err;
    ib_trx_t    trx;
    ib_id_t     table_id;

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

    // Create a database "test"
    if (ib_database_create("test") != IB_TRUE) {
        fprintf(stderr, "ib_database_create(test) failed\n");
        return 1;
    }

    ib_tbl_sch_t tbl_sch;
    // Use "test/test_table" as name
    err = ib_table_schema_create("test/test_table", &tbl_sch, IB_TBL_REDUNDANT, 0);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_table_schema_create() failed: %s\n", ib_strerror(err));
        return 1;
    }

    // Add one INT column as primary key
    // We'll keep it simple: one column 'id' INT NOT NULL
    err = ib_table_schema_add_col(tbl_sch, "id", IB_INT, IB_COL_NOT_NULL | IB_COL_UNSIGNED, 0, 4);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "adding 'id' col failed: %s\n", ib_strerror(err));
        ib_table_schema_delete(tbl_sch);
        return 1;
    }

    // Create a primary index
    ib_idx_sch_t idx_sch;
    err = ib_table_schema_add_index(tbl_sch, "PRIMARY", &idx_sch);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_table_schema_add_index(PRIMARY) failed: %s\n", ib_strerror(err));
        ib_table_schema_delete(tbl_sch);
        return 1;
    }

    // Add 'id' to the primary index
    err = ib_index_schema_add_col(idx_sch, "id", 0);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_index_schema_add_col(id) failed: %s\n", ib_strerror(err));
        ib_table_schema_delete(tbl_sch);
        return 1;
    }

    // Make it clustered
    err = ib_index_schema_set_clustered(idx_sch);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_index_schema_set_clustered() failed: %s\n", ib_strerror(err));
        ib_table_schema_delete(tbl_sch);
        return 1;
    }

    // Now create the table
    trx = ib_trx_begin(IB_TRX_REPEATABLE_READ);
    if (!trx) {
        fprintf(stderr, "ib_trx_begin() failed\n");
        ib_table_schema_delete(tbl_sch);
        return 1;
    }

    err = ib_schema_lock_exclusive(trx);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_schema_lock_exclusive() failed: %s\n", ib_strerror(err));
        ib_trx_rollback(trx);
        ib_table_schema_delete(tbl_sch);
        return 1;
    }

    err = ib_table_create(trx, tbl_sch, &table_id);
    if (err != DB_SUCCESS && err != DB_TABLE_IS_BEING_USED) {
        fprintf(stderr, "ib_table_create() failed: %s\n", ib_strerror(err));
        ib_schema_unlock(trx);
        ib_trx_rollback(trx);
        ib_table_schema_delete(tbl_sch);
        return 1;
    }

    err = ib_schema_unlock(trx);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_schema_unlock failed: %s\n", ib_strerror(err));
        ib_trx_rollback(trx);
        ib_table_schema_delete(tbl_sch);
        return 1;
    }

    err = ib_trx_commit(trx);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "commit ddl trx failed: %s\n", ib_strerror(err));
        ib_table_schema_delete(tbl_sch);
        return 1;
    }

    ib_table_schema_delete(tbl_sch);

    // Insert a single row (id=1234) to test if we can proceed
    trx = ib_trx_begin(IB_TRX_REPEATABLE_READ);
    if (!trx) {
        fprintf(stderr, "ib_trx_begin() failed for insert\n");
        return 1;
    }

    ib_crsr_t crsr;
    err = ib_cursor_open_table_using_id(table_id, trx, &crsr);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_cursor_open_table_using_id() failed: %s\n", ib_strerror(err));
        ib_trx_rollback(trx);
        return 1;
    }

    ib_tpl_t ins_tpl = ib_clust_read_tuple_create(crsr);
    if (!ins_tpl) {
        fprintf(stderr, "ib_clust_read_tuple_create failed\n");
        ib_cursor_close(crsr);
        ib_trx_rollback(trx);
        return 1;
    }

    err = ib_tuple_write_u64(ins_tpl, 0, 1234ULL);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_tuple_write_u64 failed: %s\n", ib_strerror(err));
        ib_tuple_delete(ins_tpl);
        ib_cursor_close(crsr);
        ib_trx_rollback(trx);
        return 1;
    }

    err = ib_cursor_insert_row(crsr, ins_tpl);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_cursor_insert_row() failed: %s\n", ib_strerror(err));
        ib_tuple_delete(ins_tpl);
        ib_cursor_close(crsr);
        ib_trx_rollback(trx);
        return 1;
    }

    ib_tuple_delete(ins_tpl);
    err = ib_trx_commit(trx);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "commit after insert failed: %s\n", ib_strerror(err));
    }

    ib_cursor_close(crsr);

// 6. Search for the inserted row
    trx = ib_trx_begin(IB_TRX_REPEATABLE_READ);
    ib_cursor_attach_trx(crsr, trx);
    ib_tpl_t search_tpl = ib_clust_search_tuple_create(crsr);
    ib_tuple_write_u64(search_tpl, 0, 1234ULL); // Searching by 'id'

    int result;
    err = ib_cursor_moveto(crsr, search_tpl, IB_CUR_GE, &result);
    if (err == DB_SUCCESS && result == 0 && ib_cursor_is_positioned(crsr)) {
        // We found the exact match
        ib_tpl_t read_tpl = ib_clust_read_tuple_create(crsr);
        err = ib_cursor_read_row(crsr, read_tpl);
        if (err == DB_SUCCESS) {
            char buf[256];
            ib_ulint_t len = ib_col_copy_value(read_tpl, 1, buf, sizeof(buf)-1);
            if (len != IB_SQL_NULL) {
                buf[len] = '\0';
                printf("Found row: id=1234, name=%s\n", buf);
            }
        }
        ib_tuple_delete(read_tpl);
    } else {
        printf("Row not found.\n");
    }

    // Clean up search
    ib_tuple_delete(search_tpl);

    // Commit after reading
    ib_trx_commit(trx);

    // Close cursor and shutdown
    ib_cursor_close(crsr);

    ib_shutdown(IB_SHUTDOWN_NORMAL);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_shutdown() failed: %s\n", ib_strerror(err));
    }

    return 0;
}
