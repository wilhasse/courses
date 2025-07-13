/*
  Simple MySQL chDB UDF that actually executes chDB
*/

extern "C" {

#include <mysql.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define CHDB_BINARY_PATH "/home/cslog/chdb/buildlib/programs/clickhouse"

// Global buffer for returning results
static char global_result_buffer[65535];

// Helper function to escape single quotes in SQL
static void escape_sql(char *dest, const char *src, size_t src_len, size_t max_dest) {
    size_t j = 0;
    for (size_t i = 0; i < src_len && j < max_dest - 2; i++) {
        if (src[i] == '\'') {
            if (j < max_dest - 3) {
                dest[j++] = '\'';
                dest[j++] = '\'';
            }
        } else {
            dest[j++] = src[i];
        }
    }
    dest[j] = '\0';
}

// UDF function: chdb_query(sql_string)
char *chdb_query(UDF_INIT *initid, UDF_ARGS *args, char *result,
                unsigned long *length, char *is_null, char *error) {
    *is_null = 0;
    *error = 0;
    
    if (args->arg_count != 1) {
        const char* err_msg = "ERROR: chdb_query() requires exactly one string argument";
        strcpy(global_result_buffer, err_msg);
        *length = strlen(global_result_buffer);
        return global_result_buffer;
    }
    
    // Get the query string
    char* query = args->args[0];
    if (!query || args->lengths[0] == 0) {
        const char* err_msg = "ERROR: Query string is NULL or empty";
        strcpy(global_result_buffer, err_msg);
        *length = strlen(global_result_buffer);
        return global_result_buffer;
    }
    
    // Escape the SQL query
    char escaped_query[8192];
    escape_sql(escaped_query, query, args->lengths[0], sizeof(escaped_query));
    
    // Build command to execute chDB using single quotes
    char command[10240];
    snprintf(command, sizeof(command), 
             "%s local --query '%s' --output-format TabSeparated 2>&1",
             CHDB_BINARY_PATH, escaped_query);
    
    // Execute the command
    FILE *fp = popen(command, "r");
    if (fp == NULL) {
        const char* err_msg = "ERROR: Failed to execute chDB";
        strcpy(global_result_buffer, err_msg);
        *length = strlen(global_result_buffer);
        return global_result_buffer;
    }
    
    // Read the output
    size_t total_read = 0;
    size_t buffer_size = sizeof(global_result_buffer) - 1;
    
    while (total_read < buffer_size) {
        size_t bytes_read = fread(global_result_buffer + total_read, 1, 
                                 buffer_size - total_read, fp);
        if (bytes_read == 0) break;
        total_read += bytes_read;
    }
    
    int exit_status = pclose(fp);
    
    // Check if chDB command failed
    if (exit_status != 0) {
        // The error message is already in the buffer from stderr
        global_result_buffer[total_read] = '\0';
        *length = total_read;
        return global_result_buffer;
    }
    
    // Null terminate and remove trailing newline if present
    global_result_buffer[total_read] = '\0';
    if (total_read > 0 && global_result_buffer[total_read-1] == '\n') {
        global_result_buffer[total_read-1] = '\0';
        total_read--;
    }
    
    *length = total_read;
    return global_result_buffer;
}

// UDF initialization function
bool chdb_query_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    if (args->arg_count != 1) {
        strcpy(message, "chdb_query() requires exactly one string argument");
        return true;  // error
    }
    
    // Force argument to be treated as string
    args->arg_type[0] = STRING_RESULT;
    
    // Set result properties
    initid->max_length = 65535;  // Max result length
    initid->maybe_null = 0;      // Result is never NULL
    initid->const_item = 0;      // Result is not constant
    
    // Clear the global buffer
    memset(global_result_buffer, 0, sizeof(global_result_buffer));
    
    return false;  // success
}

// UDF cleanup function
void chdb_query_deinit(UDF_INIT *initid) {
    // Nothing to clean up
}

} // extern "C"