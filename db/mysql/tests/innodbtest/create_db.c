#include <stdio.h>
#include <string.h>
#include "innodb.h"

int main() {
    ib_err_t    err;
    ib_trx_t    trx;
    ib_id_t     table_id;
    ib_tbl_sch_t tbl_sch;  // Declare the table schema handle here
    ib_crsr_t   crsr;
    ib_tpl_t    ins_tpl;
    ib_idx_sch_t idx_sch;

    // Initialize InnoDB
    err = ib_init();
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_init() failed: %s\n", ib_strerror(err));
        return 1;
    }

    // Start InnoDB
    err = ib_startup(NULL);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_startup() failed: %s\n", ib_strerror(err));
        return 1;
    }

    // Create a database "test"
    if (ib_database_create("test") != IB_TRUE) {
        fprintf(stderr, "ib_database_create(test) failed\n");
        // Even if this fails, we might still continue if database already exists
        // but let's just return for simplicity
        return 1;
    }

    // Create a table schema with IB_TBL_REDUNDANT format
    err = ib_table_schema_create("test/test_table", &tbl_sch, IB_TBL_REDUNDANT, 0);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_table_schema_create() failed: %s\n", ib_strerror(err));
        return 1;
    }

    // Add a 4-byte NOT NULL INT column named 'id'
    err = ib_table_schema_add_col(tbl_sch, "id", IB_INT, IB_COL_NOT_NULL, 0, 4);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "adding 'id' col failed: %s\n", ib_strerror(err));
        ib_table_schema_delete(tbl_sch);
        return 1;
    }

    // Add a PRIMARY index
    err = ib_table_schema_add_index(tbl_sch, "PRIMARY", &idx_sch);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_table_schema_add_index(PRIMARY) failed: %s\n", ib_strerror(err));
        ib_table_schema_delete(tbl_sch);
        return 1;
    }

    // Add 'id' column to the primary index
    err = ib_index_schema_add_col(idx_sch, "id", 0);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_index_schema_add_col(id) failed: %s\n", ib_strerror(err));
        ib_table_schema_delete(tbl_sch);
        return 1;
    }

    // Make the index clustered
    err = ib_index_schema_set_clustered(idx_sch);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_index_schema_set_clustered() failed: %s\n", ib_strerror(err));
        ib_table_schema_delete(tbl_sch);
        return 1;
    }

    // Create the table in the dictionary
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

    // Free the schema
    ib_table_schema_delete(tbl_sch);

    // Insert a row
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

    ins_tpl = ib_clust_read_tuple_create(crsr);
    if (!ins_tpl) {
        fprintf(stderr, "ib_clust_read_tuple_create failed\n");
        ib_cursor_close(crsr);
        ib_trx_rollback(trx);
        return 1;
    }

    // Write a 32-bit integer (id = 1234)
    err = ib_tuple_write_u32(ins_tpl, 0, 1234UL);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_tuple_write_u32 failed: %s\n", ib_strerror(err));
        ib_tuple_delete(ins_tpl);
        ib_cursor_close(crsr);
        ib_trx_rollback(trx);
        return 1;
    }

    // Insert the row
    err = ib_cursor_insert_row(crsr, ins_tpl);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_cursor_insert_row failed: %s\n", ib_strerror(err));
        ib_tuple_delete(ins_tpl);
        ib_cursor_close(crsr);
        ib_trx_rollback(trx);
        return 1;
    }

    ib_tuple_delete(ins_tpl);

    // Close the cursor before committing
    err = ib_cursor_close(crsr);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_cursor_close failed: %s\n", ib_strerror(err));
        ib_trx_rollback(trx);
        return 1;
    }

    // Commit the transaction so the data is persisted
    err = ib_trx_commit(trx);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "commit after insert failed: %s\n", ib_strerror(err));
        // Even if commit failed, we can still proceed to shutdown
    }

    // Finally, shutdown InnoDB gracefully
    err = ib_shutdown(IB_SHUTDOWN_NORMAL);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_shutdown() failed: %s\n", ib_strerror(err));
    }

    return 0;
}
