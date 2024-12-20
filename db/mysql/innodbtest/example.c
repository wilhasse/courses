#include <stdio.h>
#include <string.h>
#include "innodb.h"

int main() {
    ib_err_t    err;
    ib_trx_t    trx;
    ib_id_t     table_id;
    ib_crsr_t   crsr;

    // 1. Initialize InnoDB
    err = ib_init();
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_init() failed: %s\n", ib_strerror(err));
        return 1;
    }

    // Optional configuration can be done here
    // For now rely on defaults
    err = ib_startup(NULL);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_startup() failed: %s\n", ib_strerror(err));
        return 1;
    }

    // 2. Create a table schema
    ib_tbl_sch_t tbl_sch;
    err = ib_table_schema_create("test_table", &tbl_sch, IB_TBL_COMPACT, 0);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_table_schema_create() failed: %s\n", ib_strerror(err));
        return 1;
    }

    // Add columns: 'id' BIGINT UNSIGNED NOT NULL and 'name' VARCHAR(255)
    err = ib_table_schema_add_col(tbl_sch, "id", IB_INT, IB_COL_UNSIGNED | IB_COL_NOT_NULL, 0, 8);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "adding 'id' col failed: %s\n", ib_strerror(err));
        ib_table_schema_delete(tbl_sch);
        return 1;
    }

    // Use the macro ib_tbl_sch_add_varchar_col for adding a varchar column
    err = ib_tbl_sch_add_varchar_col(tbl_sch, "name", 255);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "adding 'name' col failed: %s\n", ib_strerror(err));
        ib_table_schema_delete(tbl_sch);
        return 1;
    }

    // Define a primary index on 'id'
    ib_idx_sch_t idx_sch;
    err = ib_table_schema_add_index(tbl_sch, "PRIMARY", &idx_sch);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_table_schema_add_index(PRIMARY) failed: %s\n", ib_strerror(err));
        ib_table_schema_delete(tbl_sch);
        return 1;
    }

    err = ib_index_schema_add_col(idx_sch, "id", 0);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_index_schema_add_col(id) failed: %s\n", ib_strerror(err));
        ib_table_schema_delete(tbl_sch);
        return 1;
    }

    err = ib_index_schema_set_clustered(idx_sch);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_index_schema_set_clustered() failed: %s\n", ib_strerror(err));
        ib_table_schema_delete(tbl_sch);
        return 1;
    }

    // 3. Create the table in the dictionary
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
        fprintf(stderr, "Committing DDL trx failed: %s\n", ib_strerror(err));
        ib_table_schema_delete(tbl_sch);
        return 1;
    }

    ib_table_schema_delete(tbl_sch);

    // 4. Insert a row: {id=1234, name="Alice"}
    trx = ib_trx_begin(IB_TRX_REPEATABLE_READ);
    if (!trx) {
        fprintf(stderr, "ib_trx_begin() for insert failed\n");
        return 1;
    }

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

    const char* name_val = "Alice";
    err = ib_col_set_value(ins_tpl, 1, name_val, strlen(name_val));
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_col_set_value(name) failed: %s\n", ib_strerror(err));
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
        // Not returning since we can still try to continue
    }

    // 5. Search for the inserted row by 'id' = 1234
    trx = ib_trx_begin(IB_TRX_REPEATABLE_READ);
    if (!trx) {
        fprintf(stderr, "ib_trx_begin() for search failed\n");
        // Cleanup? The DB is still open, but let's just return
        return 1;
    }

    ib_cursor_attach_trx(crsr, trx);

    ib_tpl_t search_tpl = ib_clust_search_tuple_create(crsr);
    if (!search_tpl) {
        fprintf(stderr, "ib_clust_search_tuple_create failed\n");
        ib_cursor_close(crsr);
        ib_trx_rollback(trx);
        return 1;
    }

    err = ib_tuple_write_u64(search_tpl, 0, 1234ULL);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_tuple_write_u64(search) failed: %s\n", ib_strerror(err));
        ib_tuple_delete(search_tpl);
        ib_cursor_close(crsr);
        ib_trx_rollback(trx);
        return 1;
    }

    int result;
    err = ib_cursor_moveto(crsr, search_tpl, IB_CUR_GE, &result);
    if (err == DB_SUCCESS && result == 0 && ib_cursor_is_positioned(crsr)) {
        ib_tpl_t read_tpl = ib_clust_read_tuple_create(crsr);
        if (!read_tpl) {
            fprintf(stderr, "ib_clust_read_tuple_create(read) failed\n");
            ib_tuple_delete(search_tpl);
            ib_cursor_close(crsr);
            ib_trx_rollback(trx);
            return 1;
        }

        err = ib_cursor_read_row(crsr, read_tpl);
        if (err == DB_SUCCESS) {
            char buf[256];
            ib_ulint_t len = ib_col_copy_value(read_tpl, 1, buf, sizeof(buf)-1);
            if (len != IB_SQL_NULL) {
                buf[len] = '\0';
                printf("Found row: id=1234, name=%s\n", buf);
            } else {
                printf("Found row: id=1234, name=NULL\n");
            }
        } else {
            fprintf(stderr, "ib_cursor_read_row failed: %s\n", ib_strerror(err));
        }

        ib_tuple_delete(read_tpl);
    } else {
        printf("Row not found.\n");
    }

    ib_tuple_delete(search_tpl);

    // Commit the read transaction
    err = ib_trx_commit(trx);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "commit after read failed: %s\n", ib_strerror(err));
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
