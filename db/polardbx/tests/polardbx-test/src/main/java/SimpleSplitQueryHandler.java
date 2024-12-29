import com.alibaba.druid.sql.ast.SQLStatement;
import com.alibaba.druid.sql.ast.statement.SQLSelectQueryBlock;
import com.alibaba.druid.sql.ast.statement.SQLSelectStatement;
import com.alibaba.druid.sql.dialect.mysql.parser.MySqlStatementParser;
import com.alibaba.druid.sql.parser.ParserException;
import com.alibaba.polardbx.net.buffer.ByteBufferHolder;
import com.alibaba.polardbx.net.compress.IPacketOutputProxy;
import com.alibaba.polardbx.net.compress.PacketOutputProxyFactory;
import com.alibaba.polardbx.net.packet.*;
import com.alibaba.polardbx.net.util.CharsetUtil;
import com.alibaba.polardbx.rpc.result.XResult;
import com.alibaba.polardbx.rpc.result.XResultUtil;

import java.util.ArrayList;
import java.util.List;
import java.util.TimeZone;

public class SimpleSplitQueryHandler extends SimpleQueryHandler {

    public SimpleSplitQueryHandler(DebugConnection connection) {
        super(connection);
    }

    @Override
    public void query(String sql) {
        System.out.println("Received query: " + sql);

        // 1) Parse with Alibaba Druid
        SQLStatement stmt;
        try {
            MySqlStatementParser parser = new MySqlStatementParser(sql);
            List<SQLStatement> stmtList = parser.parseStatementList();
            if (stmtList.isEmpty()) {
                sendErrorResponse("No valid statement found!");
                return;
            }
            stmt = stmtList.get(0); // assume one statement
        } catch (ParserException pe) {
            sendErrorResponse("SQL parse error: " + pe.getMessage());
            return;
        }

        // 2) Check if it's a SELECT (single table) we can chunk by PK
        if (stmt instanceof SQLSelectStatement) {
            SQLSelectStatement selectStmt = (SQLSelectStatement) stmt;
            if (canChunk(selectStmt)) {
                // 3) If it’s chunkable, build two “range queries” for demonstration
                doChunkedQuery(selectStmt);
                return;
            }
        }

        // 4) Fallback: if not chunkable, just run directly
        try {
            XResult result = connectionPool.getNextConnection().execQuery(sql);
            sendResultSetResponse(result);
        } catch (Exception e) {
            e.printStackTrace();
            sendErrorResponse("Error executing query: " + e.getMessage());
        }
    }

    protected boolean canChunk(SQLSelectStatement selectStmt) {
        // Simplify to a single SELECT query block
        if (!(selectStmt.getSelect().getQuery() instanceof SQLSelectQueryBlock)) {
            return false;
        }
        SQLSelectQueryBlock queryBlock = (SQLSelectQueryBlock) selectStmt.getSelect().getQuery();

        // Check single table
        if (queryBlock.getFrom() == null) {
            return false;
        }

        // For a real approach, you’d check if from is a single table, no subqueries, etc.
        // Hard-code that the table must be named "table1"
        String tableName = queryBlock.getFrom().toString().replace("`", "").trim();
        if (!tableName.equalsIgnoreCase("customer")) {
            return false;
        }

        // Check if ORDER BY c_name
        if (queryBlock.getOrderBy() == null || queryBlock.getOrderBy().getItems().size() != 1) {
            return false;
        }
        String orderColumn = queryBlock.getOrderBy().getItems().get(0).getExpr().toString();
        if (!orderColumn.equalsIgnoreCase("c_name")) {
            return false;
        }

        // Optionally check if c_name is the primary key
        // Hard-code in this PoC
        return true;
    }

    protected void doChunkedQuery(SQLSelectStatement originalSelect) {
        // In a real solution, you'd rewrite the AST carefully.
        // For demonstration, we just build two new strings by appending
        // the chunked conditions.

        // E.g. original:  SELECT c_name FROM customer ORDER BY c_name2
        // chunk1:         SELECT c_name FROM customer WHERE c_custkey < 10  ORDER BY c_name2
        // chunk2:         SELECT c_name FROM customer WHERE c_custkey >= 10 ORDER BY c_name2

        String sqlChunk1 = buildChunkSQL(originalSelect, "< 10");
        String sqlChunk2 = buildChunkSQL(originalSelect, ">= 10");

        System.out.println("Chunk1 SQL: " + sqlChunk1);
        System.out.println("Chunk2 SQL: " + sqlChunk2);

        try {
            // 1) Execute both queries on two separate "backends"
            //    - This example still uses the same polardbConnection, but you
            //      could have two different XConnection objects, or multiple servers
            XResult result1 = connectionPool.getNextConnection().execQuery(sqlChunk1);
            XResult result2 = connectionPool.getNextConnection().execQuery(sqlChunk1);

            // 2) Merge the results in memory (since we have an ORDER BY on col2)
            //    - We'll do a naive 2-way merge
            List<List<Object>> mergedRows = mergeOrderedResults(result1, result2);

            // 3) Convert merged rows into a "fake" XResult so we can reuse sendResultSetResponse
            //    or just write a new method to write out the final result set.
            //    For simplicity, here is a custom approach that reuses the metadata from chunk1
            sendMergedResponse(result1, mergedRows);

        } catch (Exception e) {
            e.printStackTrace();
            sendErrorResponse("Error in chunked query: " + e.getMessage());
        }
    }

    protected String buildChunkSQL(SQLSelectStatement original, String rangeCondition) {
        // Convert AST back to string, then tack on "AND c_custkey < 10" (or ">= 10")
        // A real approach would manipulate the AST instead of just string hacks.

        String originalSql = original.toString();
        // e.g. "SELECT c_name FROM customer WHERE c_custkey < 10  ORDER BY c_name2"
        // We'll do something hacky:
        // Insert "AND c_custkey < 10" right before the "ORDER BY"
        int orderByIndex = originalSql.toLowerCase().indexOf("order by");
        if (orderByIndex < 0) {
            // fallback
            return originalSql + " AND c_custkey " + rangeCondition;
        }
        String beforeOrder = originalSql.substring(0, orderByIndex).trim();
        String afterOrder = originalSql.substring(orderByIndex);

        // if there's a WHERE, append " AND c_custkey < 10"
        // if there's no WHERE, change it to "WHERE c_custkey < 10"
        if (beforeOrder.toLowerCase().contains("where")) {
            beforeOrder += " AND c_custkey " + rangeCondition + " ";
        } else {
            beforeOrder += " WHERE c_custkey " + rangeCondition + " ";
        }
        return beforeOrder + afterOrder;
    }

    protected List<List<Object>> mergeOrderedResults(XResult result1, XResult result2) throws Exception {
        // Pull row data from each result into memory (for demonstration).
        // In reality, you'd stream them or do a smarter merge.
        List<List<Object>> rows1 = readAllRows(result1);
        List<List<Object>> rows2 = readAllRows(result2);

        List<List<Object>> merged = new ArrayList<>(rows1.size() + rows2.size());
        int i = 0, j = 0;
        // assume each row has "c_custkey" at index 0 if SELECT c_custkey ...
        while (i < rows1.size() && j < rows2.size()) {
            long col2_1 = Long.parseLong(String.valueOf(rows1.get(i).get(0)));
            long col2_2 = Long.parseLong(String.valueOf(rows2.get(j).get(0)));
            if (col2_1 <= col2_2) {
                merged.add(rows1.get(i));
                i++;
            } else {
                merged.add(rows2.get(j));
                j++;
            }
        }
        // add remainder
        while (i < rows1.size()) {
            merged.add(rows1.get(i++));
        }
        while (j < rows2.size()) {
            merged.add(rows2.get(j++));
        }

        return merged;
    }

    protected List<List<Object>> readAllRows(XResult xres) throws Exception {
        List<List<Object>> rows = new ArrayList<>();
        while (xres.next() != null) {
            // each row is stored in xres.current().getRow()
            // convert to Java objects
            List<Object> oneRow = new ArrayList<>();
            for (int colIndex = 0; colIndex < xres.getMetaData().size(); colIndex++) {
                Object value = XResultUtil.resultToObject(
                        xres.getMetaData().get(colIndex),
                        xres.current().getRow().get(colIndex),
                        true,
                        TimeZone.getDefault()
                ).getKey();
                oneRow.add(value);
            }
            rows.add(oneRow);
        }
        return rows;
    }

    protected void sendMergedResponse(XResult chunk1Meta, List<List<Object>> mergedRows) {
        ByteBufferHolder buffer = null;
        try {
            byte packetId = 0;
            buffer = connection.allocate();
            buffer.clear();

            IPacketOutputProxy proxy = PacketOutputProxyFactory.getInstance().createProxy(connection, buffer);
            proxy.packetBegin();

            // 1) Write header
            ResultSetHeaderPacket header = new ResultSetHeaderPacket();
            header.packetId = ++packetId;
            header.fieldCount = chunk1Meta.getMetaData().size();
            header.write(proxy);

            // 2) Write fields
            for (int i = 0; i < chunk1Meta.getMetaData().size(); i++) {
                FieldPacket field = new FieldPacket();
                field.packetId = ++packetId;
                field.charsetIndex = CharsetUtil.getIndex("utf8");
                field.name = chunk1Meta.getMetaData().get(i).getName().toByteArray();
                field.type = convertPolarDBTypeToMySQLType(chunk1Meta.getMetaData().get(i));
                field.catalog = "def".getBytes();
                field.db = new byte[0];
                field.table = new byte[0];
                field.orgTable = new byte[0];
                field.orgName = field.name;
                field.decimals = 0;
                field.flags = 0;
                field.length = 255;
                field.write(proxy);
            }

            // 3) Write EOF
            if (!connection.isEofDeprecated()) {
                EOFPacket eof = new EOFPacket();
                eof.packetId = ++packetId;
                eof.write(proxy);
            }

            // 4) Write merged rows
            for (List<Object> rowData : mergedRows) {
                RowDataPacket row = new RowDataPacket(chunk1Meta.getMetaData().size());
                for (Object value : rowData) {
                    row.add(value != null ? value.toString().getBytes() : null);
                }
                row.packetId = ++packetId;
                row.write(proxy);
            }

            // 5) Final EOF
            EOFPacket lastEof = new EOFPacket();
            lastEof.packetId = ++packetId;
            lastEof.write(proxy);

            proxy.packetEnd();
        } catch (Exception e) {
            e.printStackTrace();
            if (buffer != null) {
                connection.recycleBuffer(buffer);
            }
        }
    }
}
