#include <stdio.h>
#include <string.h>
#include "innodb.h"

int main() {
    ib_err_t    err;
    ib_trx_t    trx;
    ib_id_t     table_id;
    ib_tbl_sch_t tbl_sch;  // Declare the table schema handle here
    ib_crsr_t   crsr;

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

    // Try creating a table with one INT column only.
    // Make sure tbl_sch is declared above
    err = ib_table_schema_create("test/test_table", &tbl_sch, IB_TBL_REDUNDANT, 0);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_table_schema_create() failed: %s\n", ib_strerror(err));
        return 1;
    }

    // Add a 4-byte NOT NULL INT column
    err = ib_table_schema_add_col(tbl_sch, "id", IB_INT, IB_COL_NOT_NULL, 0, 4);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "adding 'id' col failed: %s\n", ib_strerror(err));
        ib_table_schema_delete(tbl_sch);
        return 1;
    }

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

    // Now create the table
    trx = ib_trx_begin(IB_TRX_REPEATABLE_READ);
    ib_schema_lock_exclusive(trx);
    err = ib_table_create(trx, tbl_sch, &table_id);
    ib_schema_unlock(trx);
    ib_trx_commit(trx);
    ib_table_schema_delete(tbl_sch);

    // Insert a row with ib_tuple_write_u32 since len=4
    trx = ib_trx_begin(IB_TRX_REPEATABLE_READ);
    err = ib_cursor_open_table_using_id(table_id, trx, &crsr);

    ib_tpl_t ins_tpl = ib_clust_read_tuple_create(crsr);

    // Write a 32-bit integer since column is 4 bytes
    err = ib_tuple_write_u32(ins_tpl, 0, 1234UL);
    if (err != DB_SUCCESS) {
        fprintf(stderr, "ib_tuple_write_u32 failed: %s\n", ib_strerror(err));
        // Handle error
    }

    // Would need to insert row, commit, and shutdown...
    // This is just a snippet to show how to fix tbl_sch usage
    // Complete the rest of the logic as needed.

    return 0;
}
