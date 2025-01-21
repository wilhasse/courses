package com.alibaba.druid;
import java.util.List;

import com.alibaba.druid.sql.SQLUtils;
import com.alibaba.druid.sql.ast.SQLExpr;
import com.alibaba.druid.sql.ast.SQLOrderBy;
import com.alibaba.druid.sql.ast.SQLStatement;
import com.alibaba.druid.sql.ast.statement.SQLJoinTableSource;
import com.alibaba.druid.sql.ast.statement.SQLSelectItem;
import com.alibaba.druid.sql.ast.statement.SQLSelectQueryBlock;
import com.alibaba.druid.sql.ast.statement.SQLSelectStatement;
import com.alibaba.druid.sql.ast.statement.SQLTableSource;
import com.alibaba.druid.util.JdbcConstants;

/**
 * Hello world!
 */
public class App {

    
    public static void main(String[] args) {

        // Suppose you have a complex SQL with joins and subqueries:
        String sql = 
            """
            SELECT e.name, e.salary, o.name AS orgName
            FROM employee e
                 JOIN org o ON e.org_id = o.org_id
            WHERE e.salary > 1000
            ORDER BY e.salary
            """;

        // Convert DbType to String using name()
        String dbType = JdbcConstants.MYSQL.name();

        // Parse statements
        java.util.List<SQLStatement> stmtList = SQLUtils.parseStatements(sql, dbType);

        // For a single query, typically there's just one statement
        SQLStatement statement = stmtList.get(0);

        // Replace the old instanceof check with pattern matching
        if (statement instanceof SQLSelectStatement selectStmt) {

            // Now you have the AST for the entire SELECT
            // E.g. we can print it back out in a standardized format:
            String formatted = SQLUtils.toSQLString(selectStmt, dbType);
            System.out.println("Formatted SQL:\n" + formatted);
        }

        SQLSelectStatement selectStmt = (SQLSelectStatement) statement;

        // The core of the query
        SQLSelectQueryBlock queryBlock = selectStmt.getSelect().getQueryBlock();
        if (queryBlock != null) {

            // 1) FROM clause
            SQLTableSource from = queryBlock.getFrom();
            System.out.println("FROM clause: " + from);

            // 2) SELECT items
            List<SQLSelectItem> selectItems = queryBlock.getSelectList();
            for (SQLSelectItem item : selectItems) {
                System.out.println("Select item: " + item.toString());
            }

            // 3) WHERE
            SQLExpr whereExpr = queryBlock.getWhere();
            System.out.println("WHERE expr: " + (whereExpr == null ? "none" : whereExpr.toString()));

            // 4) ORDER BY
            SQLOrderBy orderBy = queryBlock.getOrderBy();
            if (orderBy != null) {
                System.out.println("ORDER BY: " + orderBy.toString());
            }
        } 
        
        SQLTableSource from = queryBlock.getFrom();
        if (from instanceof SQLJoinTableSource join) {
            SQLTableSource leftTable = join.getLeft();
            SQLTableSource rightTable = join.getRight();
            SQLExpr condition = join.getCondition();

            System.out.println("Join type: " + join.getJoinType());
            System.out.println("Left table: " + leftTable);
            System.out.println("Right table: " + rightTable);
            System.out.println("Join condition: " + condition);
        }        
    }
}
