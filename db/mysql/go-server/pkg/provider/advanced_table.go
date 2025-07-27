package provider

import (
	"fmt"
	"strings"

	"github.com/dolthub/go-mysql-server/sql"
	"github.com/dolthub/go-mysql-server/sql/expression"
	"mysql-server-example/pkg/storage"
)

// AdvancedTable demonstrates more sophisticated table features
type AdvancedTable struct {
	*Table // Embed basic table
}

// NewAdvancedTable creates a table with advanced features
func NewAdvancedTable(name string, schema sql.Schema, storage storage.Storage, database string) *AdvancedTable {
	return &AdvancedTable{
		Table: NewTable(name, schema, storage, database),
	}
}

// WithFilters implements sql.FilteredTable for predicate pushdown
func (t *AdvancedTable) WithFilters(ctx *sql.Context, filters []sql.Expression) sql.Table {
	return &FilteredAdvancedTable{
		AdvancedTable: t,
		filters:       filters,
	}
}

// Filters implements sql.FilteredTable
func (t *AdvancedTable) Filters() []sql.Expression {
	return nil // No filters by default
}

// Projections implements sql.ProjectedTable
func (t *AdvancedTable) Projections() []string {
	return nil // No projections by default
}

// HandledFilters implements sql.FilteredTable
func (t *AdvancedTable) HandledFilters(filters []sql.Expression) []sql.Expression {
	return nil // Return unhandled filters
}

// WithProjection implements sql.ProjectedTable for column pruning (deprecated)
func (t *AdvancedTable) WithProjection(colNames []string) sql.Table {
	return t.WithProjections(colNames)
}

// WithProjections implements sql.ProjectedTable for column pruning
func (t *AdvancedTable) WithProjections(colNames []string) sql.Table {
	// Create projected schema
	projectedSchema := make(sql.Schema, 0, len(colNames))
	for _, colName := range colNames {
		for _, col := range t.schema {
			if strings.EqualFold(col.Name, colName) {
				projectedSchema = append(projectedSchema, col)
				break
			}
		}
	}

	return &ProjectedAdvancedTable{
		AdvancedTable:    t,
		projectedSchema:  projectedSchema,
		projectedColumns: colNames,
	}
}

// FilteredAdvancedTable applies filters at storage level
type FilteredAdvancedTable struct {
	*AdvancedTable
	filters []sql.Expression
}

func (ft *FilteredAdvancedTable) PartitionRows(ctx *sql.Context, partition sql.Partition) (sql.RowIter, error) {
	// Try to convert SQL filters to storage-level filters
	storageFilters := ft.convertFiltersToStorage(ctx)
	
	if len(storageFilters) > 0 {
		// Use storage-level filtering if possible
		return ft.getFilteredRowsFromStorage(ctx, storageFilters)
	}
	
	// Fallback to in-memory filtering
	baseIter, err := ft.AdvancedTable.PartitionRows(ctx, partition)
	if err != nil {
		return nil, err
	}
	
	return &filteredRowIter{
		base:    baseIter,
		filters: ft.filters,
	}, nil
}

func (ft *FilteredAdvancedTable) String() string {
	return fmt.Sprintf("FilteredTable(%s)", ft.name)
}

// convertFiltersToStorage attempts to convert SQL expressions to storage filters
func (ft *FilteredAdvancedTable) convertFiltersToStorage(ctx *sql.Context) []storage.Filter {
	var storageFilters []storage.Filter
	
	for _, filter := range ft.filters {
		if storageFilter := ft.convertSingleFilter(filter); storageFilter != nil {
			storageFilters = append(storageFilters, *storageFilter)
		}
	}
	
	return storageFilters
}

func (ft *FilteredAdvancedTable) convertSingleFilter(expr sql.Expression) *storage.Filter {
	switch e := expr.(type) {
	case *expression.GreaterThan:
		// Handle expressions like "price > 50"
		if col, ok := e.Left().(*expression.GetField); ok {
			if literal, ok := e.Right().(*expression.Literal); ok {
				return &storage.Filter{
					Column:   col.Name(),
					Operator: storage.GreaterThan,
					Value:    literal.Value(),
				}
			}
		}
	case *expression.LessThan:
		// Handle expressions like "age < 30"
		if col, ok := e.Left().(*expression.GetField); ok {
			if literal, ok := e.Right().(*expression.Literal); ok {
				return &storage.Filter{
					Column:   col.Name(),
					Operator: storage.LessThan,
					Value:    literal.Value(),
				}
			}
		}
	case *expression.Equals:
		// Handle expressions like "status = 'active'"
		if col, ok := e.Left().(*expression.GetField); ok {
			if literal, ok := e.Right().(*expression.Literal); ok {
				return &storage.Filter{
					Column:   col.Name(),
					Operator: storage.Equals,
					Value:    literal.Value(),
				}
			}
		}
	case *expression.InTuple:
		// Handle expressions like "id IN (1, 2, 3)"
		if col, ok := e.Left().(*expression.GetField); ok {
			// Get the tuple expressions
			tuple := e.Right()
			if tupleExpr, ok := tuple.(*expression.Tuple); ok {
				values := make([]interface{}, len(tupleExpr.Children()))
				for i, val := range tupleExpr.Children() {
					if literal, ok := val.(*expression.Literal); ok {
						values[i] = literal.Value()
					} else {
						return nil // Can't convert complex expressions
					}
				}
				return &storage.Filter{
					Column:   col.Name(),
					Operator: storage.In,
					Value:    values,
				}
			}
		}
	}
	
	return nil
}

func (ft *FilteredAdvancedTable) getFilteredRowsFromStorage(ctx *sql.Context, filters []storage.Filter) (sql.RowIter, error) {
	// This would call an enhanced storage interface that supports filtering
	// For now, fall back to base implementation
	return ft.AdvancedTable.PartitionRows(ctx, &singlePartition{})
}

// ProjectedAdvancedTable only returns specified columns
type ProjectedAdvancedTable struct {
	*AdvancedTable
	projectedSchema  sql.Schema
	projectedColumns []string
}

func (pt *ProjectedAdvancedTable) Schema() sql.Schema {
	return pt.projectedSchema
}

func (pt *ProjectedAdvancedTable) PartitionRows(ctx *sql.Context, partition sql.Partition) (sql.RowIter, error) {
	baseIter, err := pt.AdvancedTable.PartitionRows(ctx, partition)
	if err != nil {
		return nil, err
	}
	
	// Create column mapping
	columnMap := make([]int, len(pt.projectedColumns))
	for i, projCol := range pt.projectedColumns {
		for j, baseCol := range pt.AdvancedTable.schema {
			if strings.EqualFold(baseCol.Name, projCol) {
				columnMap[i] = j
				break
			}
		}
	}
	
	return &projectedRowIter{
		base:      baseIter,
		columnMap: columnMap,
	}, nil
}

func (pt *ProjectedAdvancedTable) String() string {
	return fmt.Sprintf("ProjectedTable(%s, %v)", pt.name, pt.projectedColumns)
}

// filteredRowIter applies filters in memory
type filteredRowIter struct {
	base    sql.RowIter
	filters []sql.Expression
}

func (f *filteredRowIter) Next(ctx *sql.Context) (sql.Row, error) {
	for {
		row, err := f.base.Next(ctx)
		if err != nil {
			return nil, err
		}
		
		// Check all filters
		match := true
		for _, filter := range f.filters {
			result, err := filter.Eval(ctx, row)
			if err != nil {
				return nil, err
			}
			if result != true {
				match = false
				break
			}
		}
		
		if match {
			return row, nil
		}
		// Continue to next row if filters don't match
	}
}

func (f *filteredRowIter) Close(ctx *sql.Context) error {
	return f.base.Close(ctx)
}

// projectedRowIter returns only specified columns
type projectedRowIter struct {
	base      sql.RowIter
	columnMap []int
}

func (p *projectedRowIter) Next(ctx *sql.Context) (sql.Row, error) {
	baseRow, err := p.base.Next(ctx)
	if err != nil {
		return nil, err
	}
	
	projectedRow := make(sql.Row, len(p.columnMap))
	for i, colIndex := range p.columnMap {
		projectedRow[i] = baseRow[colIndex]
	}
	
	return projectedRow, nil
}

func (p *projectedRowIter) Close(ctx *sql.Context) error {
	return p.base.Close(ctx)
}

// Ensure we implement the required interfaces
var _ sql.FilteredTable = (*AdvancedTable)(nil)
var _ sql.ProjectedTable = (*AdvancedTable)(nil)
var _ sql.RowIter = (*filteredRowIter)(nil)
var _ sql.RowIter = (*projectedRowIter)(nil)