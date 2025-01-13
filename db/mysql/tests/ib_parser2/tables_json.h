#include <string>
#include <vector>
#include "tables_dict.h"

/** A minimal column-definition struct */
struct MyColumnDef {
    std::string name;            // e.g., "id", "name", ...
    std::string type_utf8;       // e.g., "int", "char", "varchar"
    uint32_t    length;          // For char(N), the N, or 4 if int
    bool        is_nullable;     // Add this
    bool        is_unsigned;     // Add this too since it's used
};

/** We store the columns here, loaded from JSON. */
static std::vector<MyColumnDef> g_columns;

int build_table_def_from_json(table_def_t* table, const char* tbl_name);

int load_ib2sdi_table_columns(const char* json_path);