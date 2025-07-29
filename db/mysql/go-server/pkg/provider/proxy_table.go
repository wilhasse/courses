package provider

import (
	"database/sql"
	"fmt"
	"io"
	"log"
	"strings"

	gmssql "github.com/dolthub/go-mysql-server/sql"
)

// ProxyTable represents a table that proxies queries to a remote MySQL database
type ProxyTable struct {
	name         string
	schema       gmssql.Schema
	remoteConn   *sql.DB
	remoteSchema string
}

// NewProxyTable creates a new proxy table
func NewProxyTable(name string, schema gmssql.Schema, conn *sql.DB, remoteSchema string) *ProxyTable {
	return &ProxyTable{
		name:         name,
		schema:       schema,
		remoteConn:   conn,
		remoteSchema: remoteSchema,
	}
}

// Name implements sql.Table
func (t *ProxyTable) Name() string {
	return t.name
}

// String implements sql.Table
func (t *ProxyTable) String() string {
	return t.name
}

// Schema implements sql.Table
func (t *ProxyTable) Schema() gmssql.Schema {
	return t.schema
}

// Collation implements sql.Table
func (t *ProxyTable) Collation() gmssql.CollationID {
	return gmssql.Collation_Default
}

// Partitions implements sql.Table
func (t *ProxyTable) Partitions(ctx *gmssql.Context) (gmssql.PartitionIter, error) {
	log.Printf("ProxyTable[%s]: Partitions() called", t.name)
	return &proxyPartitionIter{table: t.name}, nil
}

// PartitionRows implements sql.Table
func (t *ProxyTable) PartitionRows(ctx *gmssql.Context, partition gmssql.Partition) (gmssql.RowIter, error) {
	// Build SELECT query for all columns
	columns := make([]string, len(t.schema))
	for i, col := range t.schema {
		columns[i] = fmt.Sprintf("`%s`", col.Name)
	}
	
	query := fmt.Sprintf("SELECT %s FROM `%s`.`%s`", 
		strings.Join(columns, ", "), t.remoteSchema, t.name)
	
	log.Printf("ProxyTable: Executing query on remote: %s", query)
	
	rows, err := t.remoteConn.Query(query)
	if err != nil {
		return nil, fmt.Errorf("failed to query remote table: %w", err)
	}
	
	return &proxyRowIter{
		rows:   rows,
		schema: t.schema,
		table:  t.name,
	}, nil
}

// proxyPartitionIter implements sql.PartitionIter
type proxyPartitionIter struct{
	table string
	done  bool
}

func (p *proxyPartitionIter) Next(ctx *gmssql.Context) (gmssql.Partition, error) {
	if p.done {
		log.Printf("ProxyTable[%s]: PartitionIter.Next() returning EOF", p.table)
		return nil, io.EOF
	}
	p.done = true
	log.Printf("ProxyTable[%s]: PartitionIter.Next() returning partition", p.table)
	return &proxyPartition{}, nil
}

func (p *proxyPartitionIter) Close(ctx *gmssql.Context) error {
	return nil
}

// proxyPartition implements sql.Partition
type proxyPartition struct{}

func (p *proxyPartition) Key() []byte {
	return []byte("proxy")
}

// proxyRowIter implements sql.RowIter
type proxyRowIter struct {
	rows   *sql.Rows
	schema gmssql.Schema
	table  string
	count  int
}

func (r *proxyRowIter) Next(ctx *gmssql.Context) (gmssql.Row, error) {
	if !r.rows.Next() {
		if err := r.rows.Err(); err != nil {
			log.Printf("ProxyTable[%s]: Error during iteration: %v", r.table, err)
			return nil, err
		}
		log.Printf("ProxyTable[%s]: Finished reading %d rows", r.table, r.count)
		return nil, io.EOF
	}
	
	// Create scan destinations
	scanDests := make([]interface{}, len(r.schema))
	for i := range scanDests {
		scanDests[i] = new(interface{})
	}
	
	if err := r.rows.Scan(scanDests...); err != nil {
		log.Printf("ProxyTable[%s]: Scan error: %v", r.table, err)
		return nil, err
	}
	
	// Convert to gmssql.Row
	row := make(gmssql.Row, len(r.schema))
	for i, dest := range scanDests {
		val := *(dest.(*interface{}))
		if val == nil {
			row[i] = nil
			continue
		}
		
		// Convert based on column type
		row[i] = convertValue(val, r.schema[i].Type)
	}
	
	r.count++
	if r.count <= 5 { // Log first 5 rows for debugging
		log.Printf("ProxyTable[%s]: Row %d: %v", r.table, r.count, row)
	}
	
	return row, nil
}

func (r *proxyRowIter) Close(ctx *gmssql.Context) error {
	return r.rows.Close()
}

// convertValue converts a value from the MySQL driver to the expected go-mysql-server type
func convertValue(val interface{}, targetType gmssql.Type) interface{} {
	if val == nil {
		return nil
	}
	
	// Handle byte arrays (common for string types from MySQL)
	if b, ok := val.([]byte); ok {
		// Check if it's a text type by type name
		typeName := strings.ToLower(targetType.String())
		if strings.Contains(typeName, "text") || strings.Contains(typeName, "char") || strings.Contains(typeName, "string") {
			return string(b)
		}
	}
	
	// Fallback to direct return
	// Note: More sophisticated type conversion can be added here as needed
	return val
}

// Inserter implements sql.UpdatableTable
func (t *ProxyTable) Inserter(ctx *gmssql.Context) gmssql.RowInserter {
	return &proxyInserter{
		table: t,
	}
}

// Updater implements sql.UpdatableTable
func (t *ProxyTable) Updater(ctx *gmssql.Context) gmssql.RowUpdater {
	return &proxyUpdater{
		table: t,
	}
}

// Deleter implements sql.UpdatableTable
func (t *ProxyTable) Deleter(ctx *gmssql.Context) gmssql.RowDeleter {
	return &proxyDeleter{
		table: t,
	}
}

// proxyInserter implements sql.RowInserter
type proxyInserter struct {
	table *ProxyTable
}

func (i *proxyInserter) Insert(ctx *gmssql.Context, row gmssql.Row) error {
	// Build INSERT query
	columns := make([]string, len(i.table.schema))
	placeholders := make([]string, len(i.table.schema))
	values := make([]interface{}, len(i.table.schema))
	
	for idx, col := range i.table.schema {
		columns[idx] = fmt.Sprintf("`%s`", col.Name)
		placeholders[idx] = "?"
		values[idx] = row[idx]
	}
	
	query := fmt.Sprintf("INSERT INTO `%s`.`%s` (%s) VALUES (%s)",
		i.table.remoteSchema, i.table.name, 
		strings.Join(columns, ", "),
		strings.Join(placeholders, ", "))
	
	_, err := i.table.remoteConn.Exec(query, values...)
	if err != nil {
		return fmt.Errorf("failed to insert into remote table: %w", err)
	}
	
	return nil
}

func (i *proxyInserter) Close(ctx *gmssql.Context) error {
	return nil
}

func (i *proxyInserter) DiscardChanges(ctx *gmssql.Context, errorEncountered error) error {
	return nil
}

func (i *proxyInserter) StatementBegin(ctx *gmssql.Context) {
	// No-op for this implementation
}

func (i *proxyInserter) StatementComplete(ctx *gmssql.Context) error {
	return nil
}

// proxyUpdater implements sql.RowUpdater
type proxyUpdater struct {
	table *ProxyTable
}

func (u *proxyUpdater) Update(ctx *gmssql.Context, old gmssql.Row, new gmssql.Row) error {
	// Build UPDATE query using primary key
	setClauses := make([]string, 0, len(u.table.schema))
	values := make([]interface{}, 0, len(u.table.schema)+1)
	
	// Build SET clauses
	for i, col := range u.table.schema {
		if old[i] != new[i] {
			setClauses = append(setClauses, fmt.Sprintf("`%s` = ?", col.Name))
			values = append(values, new[i])
		}
	}
	
	if len(setClauses) == 0 {
		return nil // No changes
	}
	
	// Find primary key for WHERE clause
	var whereClause string
	for i, col := range u.table.schema {
		if col.PrimaryKey {
			whereClause = fmt.Sprintf("`%s` = ?", col.Name)
			values = append(values, old[i])
			break
		}
	}
	
	if whereClause == "" {
		return fmt.Errorf("cannot update without primary key")
	}
	
	query := fmt.Sprintf("UPDATE `%s`.`%s` SET %s WHERE %s",
		u.table.remoteSchema, u.table.name,
		strings.Join(setClauses, ", "),
		whereClause)
	
	_, err := u.table.remoteConn.Exec(query, values...)
	if err != nil {
		return fmt.Errorf("failed to update remote table: %w", err)
	}
	
	return nil
}

func (u *proxyUpdater) Close(ctx *gmssql.Context) error {
	return nil
}

func (u *proxyUpdater) DiscardChanges(ctx *gmssql.Context, errorEncountered error) error {
	return nil
}

func (u *proxyUpdater) StatementBegin(ctx *gmssql.Context) {
	// No-op for this implementation
}

func (u *proxyUpdater) StatementComplete(ctx *gmssql.Context) error {
	return nil
}

// proxyDeleter implements sql.RowDeleter
type proxyDeleter struct {
	table *ProxyTable
}

func (d *proxyDeleter) Delete(ctx *gmssql.Context, row gmssql.Row) error {
	// Find primary key for WHERE clause
	var whereClause string
	var pkValue interface{}
	
	for i, col := range d.table.schema {
		if col.PrimaryKey {
			whereClause = fmt.Sprintf("`%s` = ?", col.Name)
			pkValue = row[i]
			break
		}
	}
	
	if whereClause == "" {
		// If no primary key, match all columns (risky but functional)
		conditions := make([]string, 0)
		values := make([]interface{}, 0)
		for i, col := range d.table.schema {
			if row[i] == nil {
				conditions = append(conditions, fmt.Sprintf("`%s` IS NULL", col.Name))
			} else {
				conditions = append(conditions, fmt.Sprintf("`%s` = ?", col.Name))
				values = append(values, row[i])
			}
		}
		whereClause = strings.Join(conditions, " AND ")
		
		query := fmt.Sprintf("DELETE FROM `%s`.`%s` WHERE %s LIMIT 1",
			d.table.remoteSchema, d.table.name, whereClause)
		
		_, err := d.table.remoteConn.Exec(query, values...)
		return err
	}
	
	query := fmt.Sprintf("DELETE FROM `%s`.`%s` WHERE %s",
		d.table.remoteSchema, d.table.name, whereClause)
	
	_, err := d.table.remoteConn.Exec(query, pkValue)
	if err != nil {
		return fmt.Errorf("failed to delete from remote table: %w", err)
	}
	
	return nil
}

func (d *proxyDeleter) Close(ctx *gmssql.Context) error {
	return nil
}

func (d *proxyDeleter) DiscardChanges(ctx *gmssql.Context, errorEncountered error) error {
	return nil
}

func (d *proxyDeleter) StatementBegin(ctx *gmssql.Context) {
	// No-op for this implementation
}

func (d *proxyDeleter) StatementComplete(ctx *gmssql.Context) error {
	return nil
}

// Ensure ProxyTable implements required interfaces
var _ gmssql.Table = (*ProxyTable)(nil)
var _ gmssql.UpdatableTable = (*ProxyTable)(nil)
var _ gmssql.RowInserter = (*proxyInserter)(nil)
var _ gmssql.RowUpdater = (*proxyUpdater)(nil)
var _ gmssql.RowDeleter = (*proxyDeleter)(nil)