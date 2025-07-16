#!/bin/bash

# Monitor script for append mode synchronization
# This script checks the sync status between MySQL and ClickHouse

echo "=== MySQL to ClickHouse Sync Monitor ==="
echo "Date: $(date)"
echo

# Configuration
MYSQL_HOST="${MYSQL_HOST:-localhost}"
MYSQL_USER="${MYSQL_USER:-root}"
MYSQL_PASS="${MYSQL_PASS:-}"
MYSQL_DB="${MYSQL_DB:-}"
CHDB_PATH="${CHDB_PATH:-/chdb/data}"
EXECUTE_SQL="${EXECUTE_SQL:-./execute_sql}"

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Function to format numbers with commas
format_number() {
    printf "%'d" "$1"
}

# Function to check MySQL connection
check_mysql() {
    mysql -h "$MYSQL_HOST" -u "$MYSQL_USER" ${MYSQL_PASS:+-p"$MYSQL_PASS"} -e "SELECT 1" >/dev/null 2>&1
    return $?
}

# Function to check ClickHouse
check_clickhouse() {
    "$EXECUTE_SQL" -d "$CHDB_PATH" "SELECT 1" >/dev/null 2>&1
    return $?
}

# Get MySQL counts and dates
get_mysql_stats() {
    echo -e "${BLUE}MySQL Statistics:${NC}"
    
    # Get HISTORICO count and date range
    mysql_stats=$(mysql -h "$MYSQL_HOST" -u "$MYSQL_USER" ${MYSQL_PASS:+-p"$MYSQL_PASS"} "$MYSQL_DB" -sN -e "
        SELECT 
            COUNT(*) as total_rows,
            MIN(DATA) as min_date,
            MAX(DATA) as max_date
        FROM HISTORICO
    ")
    
    if [ -n "$mysql_stats" ]; then
        IFS=$'\t' read -r mysql_count mysql_min_date mysql_max_date <<< "$mysql_stats"
        echo "  HISTORICO rows: $(format_number $mysql_count)"
        echo "  Date range: $mysql_min_date to $mysql_max_date"
        
        # Get rows by month
        echo -e "\n  ${YELLOW}Rows by month:${NC}"
        mysql -h "$MYSQL_HOST" -u "$MYSQL_USER" ${MYSQL_PASS:+-p"$MYSQL_PASS"} "$MYSQL_DB" -t -e "
            SELECT 
                DATE_FORMAT(DATA, '%Y-%m') as month,
                COUNT(*) as row_count
            FROM HISTORICO
            GROUP BY DATE_FORMAT(DATA, '%Y-%m')
            ORDER BY month DESC
            LIMIT 12
        "
    else
        echo -e "  ${RED}Failed to get MySQL statistics${NC}"
        return 1
    fi
    
    # Get HISTORICO_TEXTO count
    texto_count=$(mysql -h "$MYSQL_HOST" -u "$MYSQL_USER" ${MYSQL_PASS:+-p"$MYSQL_PASS"} "$MYSQL_DB" -sN -e "SELECT COUNT(*) FROM HISTORICO_TEXTO")
    echo -e "\n  HISTORICO_TEXTO rows: $(format_number $texto_count)"
}

# Get ClickHouse counts and dates
get_clickhouse_stats() {
    echo -e "\n${BLUE}ClickHouse Statistics:${NC}"
    
    # Get HISTORICO count and date range
    ch_count=$("$EXECUTE_SQL" -d "$CHDB_PATH" "SELECT COUNT(*) FROM mysql_import.historico" 2>/dev/null | grep -oE '[0-9]+' | head -1)
    ch_min_date=$("$EXECUTE_SQL" -d "$CHDB_PATH" "SELECT MIN(data) FROM mysql_import.historico" 2>/dev/null | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}')
    ch_max_date=$("$EXECUTE_SQL" -d "$CHDB_PATH" "SELECT MAX(data) FROM mysql_import.historico" 2>/dev/null | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}')
    
    if [ -n "$ch_count" ]; then
        echo "  HISTORICO rows: $(format_number $ch_count)"
        echo "  Date range: $ch_min_date to $ch_max_date"
        
        # Get storage info
        storage_info=$("$EXECUTE_SQL" -d "$CHDB_PATH" -f TSV "
            SELECT 
                formatReadableSize(sum(bytes_on_disk)) as disk_size,
                count() as parts_count
            FROM system.parts 
            WHERE database = 'mysql_import' AND table = 'historico' AND active" 2>/dev/null | head -1)
        
        if [ -n "$storage_info" ]; then
            IFS=$'\t' read -r disk_size parts_count <<< "$storage_info"
            echo "  Storage size: $disk_size"
            echo "  Parts count: $parts_count"
        fi
    else
        echo -e "  ${RED}Failed to get ClickHouse statistics${NC}"
        return 1
    fi
    
    # Get HISTORICO_TEXTO count
    ch_texto_count=$("$EXECUTE_SQL" -d "$CHDB_PATH" "SELECT COUNT(*) FROM mysql_import.historico_texto" 2>/dev/null | grep -oE '[0-9]+' | head -1)
    echo "  HISTORICO_TEXTO rows: $(format_number ${ch_texto_count:-0})"
}

# Compare sync status
compare_sync() {
    echo -e "\n${BLUE}Sync Status:${NC}"
    
    if [ -n "$mysql_count" ] && [ -n "$ch_count" ]; then
        diff=$((mysql_count - ch_count))
        
        if [ $diff -eq 0 ]; then
            echo -e "  ${GREEN}✅ Fully synchronized${NC}"
            echo "  Both have $(format_number $mysql_count) rows"
        elif [ $diff -gt 0 ]; then
            echo -e "  ${YELLOW}⚠️  Behind by $(format_number $diff) rows${NC}"
            echo "  MySQL: $(format_number $mysql_count) | ClickHouse: $(format_number $ch_count)"
            
            # Check last sync time
            if [ -n "$ch_max_date" ] && [ -n "$mysql_max_date" ]; then
                echo -e "\n  ${YELLOW}Latest dates:${NC}"
                echo "  MySQL:      $mysql_max_date"
                echo "  ClickHouse: $ch_max_date"
                
                # Calculate time difference
                mysql_ts=$(date -d "$mysql_max_date" +%s 2>/dev/null)
                ch_ts=$(date -d "$ch_max_date" +%s 2>/dev/null)
                
                if [ -n "$mysql_ts" ] && [ -n "$ch_ts" ]; then
                    diff_seconds=$((mysql_ts - ch_ts))
                    diff_hours=$((diff_seconds / 3600))
                    diff_days=$((diff_seconds / 86400))
                    
                    if [ $diff_days -gt 0 ]; then
                        echo "  Behind by: $diff_days days"
                    elif [ $diff_hours -gt 0 ]; then
                        echo "  Behind by: $diff_hours hours"
                    else
                        echo "  Behind by: $((diff_seconds / 60)) minutes"
                    fi
                fi
            fi
            
            # Show new rows that need syncing
            echo -e "\n  ${YELLOW}Rows to sync:${NC}"
            if [ -n "$ch_max_date" ]; then
                new_rows=$(mysql -h "$MYSQL_HOST" -u "$MYSQL_USER" ${MYSQL_PASS:+-p"$MYSQL_PASS"} "$MYSQL_DB" -sN -e "
                    SELECT COUNT(*) FROM HISTORICO WHERE DATA > '$ch_max_date'
                ")
                echo "  New rows since $ch_max_date: $(format_number ${new_rows:-0})"
            fi
        else
            echo -e "  ${RED}❌ ClickHouse has more rows than MySQL!${NC}"
            echo "  MySQL: $(format_number $mysql_count) | ClickHouse: $(format_number $ch_count)"
            echo "  This should not happen - investigate data integrity"
        fi
    fi
}

# Show recent activity
show_recent_activity() {
    echo -e "\n${BLUE}Recent Activity (last 24 hours):${NC}"
    
    # MySQL recent rows
    mysql_recent=$(mysql -h "$MYSQL_HOST" -u "$MYSQL_USER" ${MYSQL_PASS:+-p"$MYSQL_PASS"} "$MYSQL_DB" -sN -e "
        SELECT COUNT(*) FROM HISTORICO 
        WHERE DATA >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
    ")
    
    # ClickHouse recent rows  
    ch_recent=$("$EXECUTE_SQL" -d "$CHDB_PATH" "
        SELECT COUNT(*) FROM mysql_import.historico 
        WHERE data >= now() - INTERVAL 24 HOUR" 2>/dev/null | grep -oE '[0-9]+' | head -1)
    
    echo "  MySQL new rows (24h): $(format_number ${mysql_recent:-0})"
    echo "  ClickHouse new rows (24h): $(format_number ${ch_recent:-0})"
    
    # Show hourly breakdown for last 6 hours
    echo -e "\n  ${YELLOW}Hourly breakdown (last 6 hours):${NC}"
    mysql -h "$MYSQL_HOST" -u "$MYSQL_USER" ${MYSQL_PASS:+-p"$MYSQL_PASS"} "$MYSQL_DB" -t -e "
        SELECT 
            DATE_FORMAT(DATA, '%Y-%m-%d %H:00') as hour,
            COUNT(*) as new_rows
        FROM HISTORICO
        WHERE DATA >= DATE_SUB(NOW(), INTERVAL 6 HOUR)
        GROUP BY DATE_FORMAT(DATA, '%Y-%m-%d %H:00')
        ORDER BY hour DESC
    "
}

# Suggest sync command
suggest_sync() {
    if [ $diff -gt 0 ]; then
        echo -e "\n${BLUE}Sync Recommendation:${NC}"
        echo "To sync the missing data, run:"
        echo
        echo -e "${GREEN}./historico_loader_go \\"
        echo "    -host $MYSQL_HOST \\"
        echo "    -user $MYSQL_USER \\"
        if [ -n "$MYSQL_PASS" ]; then
            echo "    -password '***' \\"
        fi
        echo "    -database $MYSQL_DB \\"
        echo "    -chdb-path $CHDB_PATH \\"
        echo -e "    -append${NC}"
        echo
        echo "Estimated rows to sync: $(format_number $diff)"
        
        # Estimate time based on 30k rows/sec
        est_seconds=$((diff / 30000))
        if [ $est_seconds -gt 60 ]; then
            echo "Estimated sync time: $((est_seconds / 60)) minutes"
        else
            echo "Estimated sync time: <1 minute"
        fi
    fi
}

# Main monitoring flow
main() {
    # Validate inputs
    if [ -z "$MYSQL_DB" ]; then
        echo -e "${RED}Error: MYSQL_DB environment variable not set${NC}"
        echo "Usage: MYSQL_DB=your_database $0"
        exit 1
    fi
    
    # Check connections
    echo -n "Checking MySQL connection... "
    if check_mysql; then
        echo -e "${GREEN}✅ Connected${NC}"
    else
        echo -e "${RED}❌ Failed${NC}"
        exit 1
    fi
    
    echo -n "Checking ClickHouse connection... "
    if check_clickhouse; then
        echo -e "${GREEN}✅ Connected${NC}"
    else
        echo -e "${RED}❌ Failed${NC}"
        echo "Make sure execute_sql is built and chDB is installed"
        exit 1
    fi
    
    echo
    
    # Get statistics
    get_mysql_stats
    get_clickhouse_stats
    compare_sync
    show_recent_activity
    suggest_sync
    
    echo -e "\n${BLUE}=== Monitor completed at $(date) ===${NC}"
}

# Run main
main "$@"