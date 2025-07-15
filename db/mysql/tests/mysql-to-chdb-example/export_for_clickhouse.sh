#!/bin/bash

# MySQL connection parameters
MYSQL_HOST="$1"
MYSQL_USER="$2"
MYSQL_PASSWORD="$3"
MYSQL_DATABASE="$4"

if [ $# -lt 4 ]; then
    echo "Usage: $0 <mysql_host> <mysql_user> <mysql_password> <mysql_database>"
    echo ""
    echo "This script exports HISTORICO data from MySQL and provides ClickHouse import commands"
    exit 1
fi

echo "Exporting HISTORICO data from MySQL..."

# Export HISTORICO table
mysql -h "$MYSQL_HOST" -u "$MYSQL_USER" -p"$MYSQL_PASSWORD" "$MYSQL_DATABASE" \
    -e "SELECT ID_CONTR, SEQ, ID_FUNCIONARIO, ID_TEL, DATE_FORMAT(DATA, '%Y-%m-%d %H:%i:%s'), CODIGO, MODO FROM HISTORICO" \
    --batch --raw > historico.tsv

if [ $? -eq 0 ]; then
    echo "Export completed successfully!"
    echo "File created: historico.tsv"
    echo ""
    echo "Row count:"
    wc -l historico.tsv
    echo ""
    echo "To import into ClickHouse, use the official clickhouse-client:"
    echo ""
    echo "1. First, create the database and table:"
    echo "   clickhouse-client --query=\"CREATE DATABASE IF NOT EXISTS mysql_import\""
    echo ""
    echo "   clickhouse-client --query=\"CREATE TABLE IF NOT EXISTS mysql_import.historico ("
    echo "       id_contr Int32,"
    echo "       seq UInt16,"
    echo "       id_funcionario Int32,"
    echo "       id_tel Int32,"
    echo "       data DateTime,"
    echo "       codigo UInt16,"
    echo "       modo String"
    echo "   ) ENGINE = MergeTree() ORDER BY (id_contr, seq)\""
    echo ""
    echo "2. Then import the data:"
    echo "   clickhouse-client --query=\"INSERT INTO mysql_import.historico FROM INFILE 'historico.tsv' FORMAT TSV\""
    echo ""
    echo "Or using cat:"
    echo "   cat historico.tsv | clickhouse-client --query=\"INSERT INTO mysql_import.historico FORMAT TSV\""
else
    echo "Export failed!"
    exit 1
fi