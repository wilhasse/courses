package provider

import (
	"io"

	"github.com/dolthub/go-mysql-server/sql"
	"mysql-server-example/pkg/storage"
)

// Table implements sql.Table and sql.UpdatableTable
type Table struct {
	name     string
	schema   sql.Schema
	storage  storage.Storage
	database string
}

// NewTable creates a new table
func NewTable(name string, schema sql.Schema, storage storage.Storage, database string) *Table {
	return &Table{
		name:     name,
		schema:   schema,
		storage:  storage,
		database: database,
	}
}

// Name implements sql.Table
func (t *Table) Name() string {
	return t.name
}

// String implements sql.Table
func (t *Table) String() string {
	return t.name
}

// Schema implements sql.Table
func (t *Table) Schema() sql.Schema {
	return t.schema
}

// Collation implements sql.Table
func (t *Table) Collation() sql.CollationID {
	return sql.Collation_Default
}

// Partitions implements sql.Table
func (t *Table) Partitions(ctx *sql.Context) (sql.PartitionIter, error) {
	return &singlePartitionIter{}, nil
}

// PartitionRows implements sql.Table
func (t *Table) PartitionRows(ctx *sql.Context, partition sql.Partition) (sql.RowIter, error) {
	rows, err := t.storage.GetRows(t.database, t.name)
	if err != nil {
		return nil, err
	}
	return &tableRowIter{rows: rows, index: 0}, nil
}

// Inserter implements sql.UpdatableTable
func (t *Table) Inserter(ctx *sql.Context) sql.RowInserter {
	return &tableInserter{
		table:    t,
		storage:  t.storage,
		database: t.database,
	}
}

// Updater implements sql.UpdatableTable
func (t *Table) Updater(ctx *sql.Context) sql.RowUpdater {
	return &tableUpdater{
		table:    t,
		storage:  t.storage,
		database: t.database,
	}
}

// Deleter implements sql.UpdatableTable
func (t *Table) Deleter(ctx *sql.Context) sql.RowDeleter {
	return &tableDeleter{
		table:    t,
		storage:  t.storage,
		database: t.database,
	}
}

// singlePartitionIter implements sql.PartitionIter for a single partition
type singlePartitionIter struct {
	returned bool
}

func (s *singlePartitionIter) Next(ctx *sql.Context) (sql.Partition, error) {
	if s.returned {
		return nil, io.EOF
	}
	s.returned = true
	return &singlePartition{}, nil
}

func (s *singlePartitionIter) Close(ctx *sql.Context) error {
	return nil
}

// singlePartition implements sql.Partition
type singlePartition struct{}

func (s *singlePartition) Key() []byte {
	return []byte("single")
}

// tableRowIter implements sql.RowIter
type tableRowIter struct {
	rows  []sql.Row
	index int
}

func (t *tableRowIter) Next(ctx *sql.Context) (sql.Row, error) {
	if t.index >= len(t.rows) {
		return nil, io.EOF
	}
	row := t.rows[t.index]
	t.index++
	return row, nil
}

func (t *tableRowIter) Close(ctx *sql.Context) error {
	return nil
}

// tableInserter implements sql.RowInserter
type tableInserter struct {
	table    *Table
	storage  storage.Storage
	database string
}

func (t *tableInserter) Insert(ctx *sql.Context, row sql.Row) error {
	return t.storage.InsertRow(t.database, t.table.name, row)
}

func (t *tableInserter) Close(ctx *sql.Context) error {
	return nil
}

func (t *tableInserter) DiscardChanges(ctx *sql.Context, errorEncountered error) error {
	return nil
}

func (t *tableInserter) StatementBegin(ctx *sql.Context) {
	// No-op for this implementation
}

func (t *tableInserter) StatementComplete(ctx *sql.Context) error {
	return nil
}

// tableUpdater implements sql.RowUpdater
type tableUpdater struct {
	table    *Table
	storage  storage.Storage
	database string
}

func (t *tableUpdater) Update(ctx *sql.Context, old sql.Row, new sql.Row) error {
	return t.storage.UpdateRow(t.database, t.table.name, old, new)
}

func (t *tableUpdater) Close(ctx *sql.Context) error {
	return nil
}

func (t *tableUpdater) DiscardChanges(ctx *sql.Context, errorEncountered error) error {
	return nil
}

func (t *tableUpdater) StatementBegin(ctx *sql.Context) {
	// No-op for this implementation
}

func (t *tableUpdater) StatementComplete(ctx *sql.Context) error {
	return nil
}

// tableDeleter implements sql.RowDeleter
type tableDeleter struct {
	table    *Table
	storage  storage.Storage
	database string
}

func (t *tableDeleter) Delete(ctx *sql.Context, row sql.Row) error {
	return t.storage.DeleteRow(t.database, t.table.name, row)
}

func (t *tableDeleter) Close(ctx *sql.Context) error {
	return nil
}

func (t *tableDeleter) DiscardChanges(ctx *sql.Context, errorEncountered error) error {
	return nil
}

func (t *tableDeleter) StatementBegin(ctx *sql.Context) {
	// No-op for this implementation
}

func (t *tableDeleter) StatementComplete(ctx *sql.Context) error {
	return nil
}

// Ensure we implement the required interfaces
var _ sql.Table = (*Table)(nil)
var _ sql.UpdatableTable = (*Table)(nil)
var _ sql.PartitionIter = (*singlePartitionIter)(nil)
var _ sql.Partition = (*singlePartition)(nil)
var _ sql.RowIter = (*tableRowIter)(nil)
var _ sql.RowInserter = (*tableInserter)(nil)
var _ sql.RowUpdater = (*tableUpdater)(nil)
var _ sql.RowDeleter = (*tableDeleter)(nil)