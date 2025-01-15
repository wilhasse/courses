#include <iostream>
#include <cstdint>
#include <vector>
#include <string>
#include <cstdio>
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/istreamwrapper.h>
#include <fstream>
#include <iostream>

#include "tables_dict.h"
#include "tables_json.h"

int build_table_def_from_json(table_def_t* table, const char* tbl_name)
{
    // 1) Zero out "table_def_t"
    std::memset(table, 0, sizeof(table_def_t));

    // 2) Copy the table name
    table->name = strdup(tbl_name);

    // 3) Loop over columns
    unsigned colcount = 0;
    for (size_t i = 0; i < g_columns.size(); i++) {
        if (colcount >= MAX_TABLE_FIELDS) {
            fprintf(stderr, "[Error] Too many columns (>MAX_TABLE_FIELDS)\n");
            return 1;
        }

        field_def_t* fld = &table->fields[colcount];
        std::memset(fld, 0, sizeof(*fld));

        // (A) Name
        fld->name = strdup(g_columns[i].name.c_str());

        // (B) is_nullable => can_be_null
        // If the JSON had is_nullable => g_columns[i].nullable, adapt.
        // Let's assume we store is_nullable in g_columns[i].is_nullable:
        fld->can_be_null = g_columns[i].is_nullable;

        // (C) If the JSON had "is_unsigned" => store or adapt type
        // For example, if we see "int" + is_unsigned => FT_UINT
        bool is_unsigned = g_columns[i].is_unsigned; // e.g. from JSON

        // (D) "type_utf8" => decide the main field type
        if (g_columns[i].type_utf8.find("int") != std::string::npos) {
            if (is_unsigned) {
                fld->type = FT_UINT;
            } else {
                fld->type = FT_INT;
            }
            // 4 bytes typical
            fld->fixed_length = 4;
            // For check_fields_sizes => maybe min_length=4, max_length=4
            fld->min_length = 4;
            fld->max_length = 4;

        } else if (g_columns[i].type_utf8.find("char") != std::string::npos) {
            // Suppose "char(N)" => fixed_length = N
            // or "varchar(N)" => fixed_length=0, max_length=N
            // For a simplistic approach, do:
            fld->type = FT_CHAR;
            // if we see something like "char(1)", we do:
            fld->fixed_length = 0; // treat as variable
            fld->min_length = 0;
            fld->max_length = g_columns[i].length; 
            // (If your code wants a truly fixed CHAR(1), you can do that logic.)

        } else if (g_columns[i].type_utf8.find("datetime") != std::string::npos) {
            // Usually datetime is 5 or 8 bytes in "undrop" logic
            fld->type         = FT_DATETIME;
            fld->fixed_length = 5; // or 8
            // min_length=5, max_length=5 for check_fields_sizes
            fld->min_length   = 5;
            fld->max_length   = 5;

        } else {
            // fallback => treat as text
            fld->type = FT_TEXT;
            fld->fixed_length = 0; 
            fld->min_length   = 0;
            // if JSON has "char_length" => we do
            fld->max_length   = g_columns[i].length;
        }

        // (E) Possibly parse numeric precision, scale => decimal_digits, etc.
        // (F) Possibly parse "char_length" => do above or below.

        // done
        colcount++;
    }

    // 5) fields_count
    table->fields_count = colcount;

    // 6) Optionally compute table->n_nullable
    table->n_nullable = 0;
    for (unsigned i = 0; i < colcount; i++) {
        if (table->fields[i].can_be_null) {
            table->n_nullable++;
        }
    }

    // optionally set data_max_size, data_min_size
    // or do so in your calling code if you want consistent row checks.

    return 0;
}

/**
 * load_ib2sdi_table_columns():
 *   Parses an ib2sdi-generated JSON file (like the one you pasted),
 *   searches for the array element that has "dd_object_type" == "Table",
 *   then extracts its "columns" array from "dd_object".
 *
 * Returns 0 on success, non-0 on error.
 */
int load_ib2sdi_table_columns(const char* json_path)
{
    // 1) Open the file
    std::ifstream ifs(json_path);
    if (!ifs.is_open()) {
        std::cerr << "[Error] Could not open JSON file: " << json_path << std::endl;
        return 1;
    }

    // 2) Parse the top-level JSON
    rapidjson::IStreamWrapper isw(ifs);
    rapidjson::Document d;
    d.ParseStream(isw);
    if (d.HasParseError()) {
        std::cerr << "[Error] JSON parse error: " 
                  << rapidjson::GetParseError_En(d.GetParseError())
                  << " at offset " << d.GetErrorOffset() << std::endl;
        return 1;
    }

    if (!d.IsArray()) {
        std::cerr << "[Error] Top-level JSON is not an array.\n";
        return 1;
    }

    // 3) Find the array element whose "dd_object_type" == "Table".
    //    In your example, you had something like:
    //    [
    //       "ibd2sdi",
    //       { "type":1, "object": { "dd_object_type":"Table", ... } },
    //       { "type":2, "object": { "dd_object_type":"Tablespace", ... } }
    //    ]
    // We'll loop the array to find the "Table" entry.

    const rapidjson::Value* table_obj = nullptr;

    for (auto& elem : d.GetArray()) {
        // Each elem might be "ibd2sdi" (string) or an object with { "type":..., "object":... }
        if (elem.IsObject() && elem.HasMember("object")) {
            const rapidjson::Value& obj = elem["object"];
            if (obj.HasMember("dd_object_type") && obj["dd_object_type"].IsString()) {
                std::string ddtype = obj["dd_object_type"].GetString();
                if (ddtype == "Table") {
                    // Found the table element
                    table_obj = &obj;
                    break; 
                }
            }
        }
    }

    if (!table_obj) {
        std::cerr << "[Error] Could not find any array element with dd_object_type=='Table'.\n";
        return 1;
    }

    // 4) Inside that "object", we want "dd_object" => "columns"
    //    i.e. table_obj->HasMember("dd_object") => columns in table_obj["dd_object"]["columns"]
    if (!table_obj->HasMember("dd_object")) {
        std::cerr << "[Error] Table object is missing 'dd_object' member.\n";
        return 1;
    }
    const rapidjson::Value& dd_obj = (*table_obj)["dd_object"];

    if (!dd_obj.HasMember("columns") || !dd_obj["columns"].IsArray()) {
        std::cerr << "[Error] 'dd_object' is missing 'columns' array.\n";
        return 1;
    }

    const rapidjson::Value& columns = dd_obj["columns"];
    g_columns.clear();

    // 5) Iterate the columns array
    for (auto& c : columns.GetArray()) {
        // We expect "name", "column_type_utf8", "char_length" in each
        if (!c.HasMember("name") || !c.HasMember("column_type_utf8") || !c.HasMember("char_length")) {
            // Some columns might be hidden or missing fields
            // That's typical for DB_TRX_ID, DB_ROLL_PTR, etc.
            // We'll just skip them or give defaults.
            // For demo, skip if missing 'name' or 'column_type_utf8'
            if (!c.HasMember("name") || !c.HasMember("column_type_utf8")) {
                std::cerr << "[Warn] A column is missing 'name' or 'column_type_utf8'. Skipping.\n";
                continue;
            }
        }

        MyColumnDef def;
        def.name      = c["name"].GetString();
        def.type_utf8 = c["column_type_utf8"].GetString();

        // default length = 4 if "int"? 
        // or from "char_length"
        uint32_t length = 4; // fallback
        if (c.HasMember("char_length") && c["char_length"].IsUint()) {
            length = c["char_length"].GetUint();
        }
        def.length = length;

        // Add to global vector
        g_columns.push_back(def);

        // Optional debug
        std::cout << "[Debug] Added column: name='" << def.name
                  << "', type='" << def.type_utf8
                  << "', length=" << def.length << "\n";
    }

    ifs.close();
    return 0;
}
