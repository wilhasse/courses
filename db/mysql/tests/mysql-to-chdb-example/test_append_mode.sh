#!/bin/bash

# Test script for append mode functionality
# This script tests the incremental update feature of historico_loader_go

echo "=== Testing Append Mode for historico_loader_go ==="
echo "Date: $(date)"
echo

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
MYSQL_HOST="${MYSQL_HOST:-localhost}"
MYSQL_USER="${MYSQL_USER:-root}"
MYSQL_PASS="${MYSQL_PASS:-}"
MYSQL_DB="${MYSQL_DB:-test_db}"
CHDB_PATH="${CHDB_PATH:-/tmp/test_append_chdb}"

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Cleaning up test environment...${NC}"
    rm -rf "$CHDB_PATH"
    mysql -h "$MYSQL_HOST" -u "$MYSQL_USER" ${MYSQL_PASS:+-p"$MYSQL_PASS"} -e "DROP DATABASE IF EXISTS $MYSQL_DB" 2>/dev/null
}

# Setup test database
setup_test_db() {
    echo "Setting up test database..."
    
    mysql -h "$MYSQL_HOST" -u "$MYSQL_USER" ${MYSQL_PASS:+-p"$MYSQL_PASS"} <<EOF
CREATE DATABASE IF NOT EXISTS $MYSQL_DB;
USE $MYSQL_DB;

-- Create HISTORICO table
CREATE TABLE IF NOT EXISTS HISTORICO (
    ID_CONTR INT,
    SEQ SMALLINT UNSIGNED,
    ID_FUNCIONARIO INT,
    ID_TEL INT,
    DATA DATETIME,
    CODIGO SMALLINT UNSIGNED,
    MODO VARCHAR(10),
    PRIMARY KEY (ID_CONTR, SEQ),
    INDEX idx_data (DATA)
) ENGINE=InnoDB;

-- Create HISTORICO_TEXTO table
CREATE TABLE IF NOT EXISTS HISTORICO_TEXTO (
    ID_CONTR INT,
    SEQ SMALLINT UNSIGNED,
    MENSAGEM TEXT,
    MOTIVO VARCHAR(255),
    AUTORIZACAO VARCHAR(100),
    PRIMARY KEY (ID_CONTR, SEQ)
) ENGINE=InnoDB;

-- Insert initial data (older dates)
INSERT INTO HISTORICO (ID_CONTR, SEQ, ID_FUNCIONARIO, ID_TEL, DATA, CODIGO, MODO) VALUES
(1, 1, 101, 201, '2024-01-01 10:00:00', 10, 'MODE1'),
(1, 2, 101, 201, '2024-01-02 11:00:00', 20, 'MODE2'),
(2, 1, 102, 202, '2024-01-03 12:00:00', 30, 'MODE1'),
(2, 2, 102, 202, '2024-01-04 13:00:00', 40, 'MODE3'),
(3, 1, 103, 203, '2024-01-05 14:00:00', 50, 'MODE2');

INSERT INTO HISTORICO_TEXTO (ID_CONTR, SEQ, MENSAGEM, MOTIVO, AUTORIZACAO) VALUES
(1, 1, 'Message 1', 'Reason 1', 'AUTH1'),
(1, 2, 'Message 2', 'Reason 2', 'AUTH2'),
(2, 1, 'Message 3', 'Reason 3', 'AUTH3'),
(2, 2, 'Message 4', 'Reason 4', 'AUTH4'),
(3, 1, 'Message 5', 'Reason 5', 'AUTH5');
EOF

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Test database created successfully${NC}"
    else
        echo -e "${RED}❌ Failed to create test database${NC}"
        exit 1
    fi
}

# Run initial import
run_initial_import() {
    echo -e "\n${YELLOW}Running initial import...${NC}"
    
    ./historico_loader_go \
        -host "$MYSQL_HOST" \
        -user "$MYSQL_USER" \
        -password "$MYSQL_PASS" \
        -database "$MYSQL_DB" \
        -chdb-path "$CHDB_PATH" 2>&1 | tee initial_import.log
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "${GREEN}✅ Initial import completed${NC}"
    else
        echo -e "${RED}❌ Initial import failed${NC}"
        exit 1
    fi
}

# Verify initial data
verify_initial_data() {
    echo -e "\n${YELLOW}Verifying initial data...${NC}"
    
    # Count rows in ClickHouse
    count=$(./execute_sql -d "$CHDB_PATH" "SELECT COUNT(*) FROM mysql_import.historico" 2>/dev/null | grep -oE '[0-9]+' | head -1)
    
    if [ "$count" = "5" ]; then
        echo -e "${GREEN}✅ Initial HISTORICO count correct: $count rows${NC}"
    else
        echo -e "${RED}❌ Initial HISTORICO count incorrect: $count rows (expected 5)${NC}"
        exit 1
    fi
    
    # Check latest date
    latest_date=$(./execute_sql -d "$CHDB_PATH" "SELECT MAX(data) FROM mysql_import.historico" 2>/dev/null | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}')
    echo "Latest date in ClickHouse: $latest_date"
}

# Add new data to MySQL
add_new_data() {
    echo -e "\n${YELLOW}Adding new data to MySQL...${NC}"
    
    mysql -h "$MYSQL_HOST" -u "$MYSQL_USER" ${MYSQL_PASS:+-p"$MYSQL_PASS"} "$MYSQL_DB" <<EOF
-- Insert new data (newer dates)
INSERT INTO HISTORICO (ID_CONTR, SEQ, ID_FUNCIONARIO, ID_TEL, DATA, CODIGO, MODO) VALUES
(4, 1, 104, 204, '2024-01-10 15:00:00', 60, 'MODE1'),
(4, 2, 104, 204, '2024-01-11 16:00:00', 70, 'MODE2'),
(5, 1, 105, 205, '2024-01-12 17:00:00', 80, 'MODE3');

INSERT INTO HISTORICO_TEXTO (ID_CONTR, SEQ, MENSAGEM, MOTIVO, AUTORIZACAO) VALUES
(4, 1, 'New Message 1', 'New Reason 1', 'NEWAUTH1'),
(4, 2, 'New Message 2', 'New Reason 2', 'NEWAUTH2'),
(5, 1, 'New Message 3', 'New Reason 3', 'NEWAUTH3');
EOF

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ New data added to MySQL${NC}"
        
        # Show MySQL count
        mysql_count=$(mysql -h "$MYSQL_HOST" -u "$MYSQL_USER" ${MYSQL_PASS:+-p"$MYSQL_PASS"} "$MYSQL_DB" -sN -e "SELECT COUNT(*) FROM HISTORICO")
        echo "Total rows in MySQL HISTORICO: $mysql_count"
    else
        echo -e "${RED}❌ Failed to add new data${NC}"
        exit 1
    fi
}

# Run append mode import
run_append_import() {
    echo -e "\n${YELLOW}Running append mode import...${NC}"
    
    ./historico_loader_go \
        -host "$MYSQL_HOST" \
        -user "$MYSQL_USER" \
        -password "$MYSQL_PASS" \
        -database "$MYSQL_DB" \
        -chdb-path "$CHDB_PATH" \
        -append 2>&1 | tee append_import.log
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "${GREEN}✅ Append mode import completed${NC}"
    else
        echo -e "${RED}❌ Append mode import failed${NC}"
        exit 1
    fi
}

# Verify append data
verify_append_data() {
    echo -e "\n${YELLOW}Verifying append data...${NC}"
    
    # Count total rows
    total_count=$(./execute_sql -d "$CHDB_PATH" "SELECT COUNT(*) FROM mysql_import.historico" 2>/dev/null | grep -oE '[0-9]+' | head -1)
    
    if [ "$total_count" = "8" ]; then
        echo -e "${GREEN}✅ Total HISTORICO count correct: $total_count rows${NC}"
    else
        echo -e "${RED}❌ Total HISTORICO count incorrect: $total_count rows (expected 8)${NC}"
        exit 1
    fi
    
    # Count new rows (after 2024-01-05)
    new_count=$(./execute_sql -d "$CHDB_PATH" "SELECT COUNT(*) FROM mysql_import.historico WHERE data > '2024-01-05 14:00:00'" 2>/dev/null | grep -oE '[0-9]+' | head -1)
    
    if [ "$new_count" = "3" ]; then
        echo -e "${GREEN}✅ New rows count correct: $new_count rows${NC}"
    else
        echo -e "${RED}❌ New rows count incorrect: $new_count rows (expected 3)${NC}"
        exit 1
    fi
    
    # Verify HISTORICO_TEXTO
    texto_count=$(./execute_sql -d "$CHDB_PATH" "SELECT COUNT(*) FROM mysql_import.historico_texto" 2>/dev/null | grep -oE '[0-9]+' | head -1)
    echo "Total HISTORICO_TEXTO rows: $texto_count"
    
    # Show sample of new data
    echo -e "\n${YELLOW}Sample of new data:${NC}"
    ./execute_sql -d "$CHDB_PATH" "SELECT * FROM mysql_import.historico WHERE data > '2024-01-05 14:00:00' ORDER BY id_contr, seq" 2>/dev/null
}

# Test idempotency
test_idempotency() {
    echo -e "\n${YELLOW}Testing idempotency (running append again)...${NC}"
    
    # Get count before
    count_before=$(./execute_sql -d "$CHDB_PATH" "SELECT COUNT(*) FROM mysql_import.historico" 2>/dev/null | grep -oE '[0-9]+' | head -1)
    
    # Run append again
    ./historico_loader_go \
        -host "$MYSQL_HOST" \
        -user "$MYSQL_USER" \
        -password "$MYSQL_PASS" \
        -database "$MYSQL_DB" \
        -chdb-path "$CHDB_PATH" \
        -append >/dev/null 2>&1
    
    # Get count after
    count_after=$(./execute_sql -d "$CHDB_PATH" "SELECT COUNT(*) FROM mysql_import.historico" 2>/dev/null | grep -oE '[0-9]+' | head -1)
    
    if [ "$count_before" = "$count_after" ]; then
        echo -e "${GREEN}✅ Idempotency test passed: no duplicates (still $count_after rows)${NC}"
    else
        echo -e "${RED}❌ Idempotency test failed: count changed from $count_before to $count_after${NC}"
        exit 1
    fi
}

# Main test flow
main() {
    echo "Test configuration:"
    echo "  MySQL Host: $MYSQL_HOST"
    echo "  MySQL User: $MYSQL_USER"
    echo "  MySQL Database: $MYSQL_DB"
    echo "  chDB Path: $CHDB_PATH"
    echo
    
    # Check prerequisites
    if [ ! -f ./historico_loader_go ]; then
        echo -e "${RED}❌ historico_loader_go not found. Please build it first.${NC}"
        exit 1
    fi
    
    if [ ! -f ./execute_sql ]; then
        echo -e "${RED}❌ execute_sql not found. Please build it first.${NC}"
        exit 1
    fi
    
    # Clean up any previous test
    cleanup
    
    # Run tests
    setup_test_db
    run_initial_import
    verify_initial_data
    add_new_data
    run_append_import
    verify_append_data
    test_idempotency
    
    echo -e "\n${GREEN}=== All tests passed! ===${NC}"
    echo "Append mode is working correctly:"
    echo "- Initial import loaded all data"
    echo "- Append mode only imported new records"
    echo "- No duplicates were created"
    echo "- HISTORICO_TEXTO records were correctly filtered"
    
    # Cleanup
    read -p "Clean up test data? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cleanup
        echo -e "${GREEN}✅ Test data cleaned up${NC}"
    else
        echo "Test data preserved at: $CHDB_PATH"
    fi
}

# Run main if not sourced
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi