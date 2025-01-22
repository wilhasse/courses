import com.alibaba.druid.sql.ast.SQLStatement;
import com.alibaba.druid.sql.ast.statement.SQLSelectQueryBlock;
import com.alibaba.druid.sql.ast.statement.SQLSelectStatement;
import com.alibaba.druid.sql.dialect.mysql.parser.MySqlStatementParser;
import com.alibaba.druid.sql.parser.ParserException;
import com.alibaba.polardbx.Fields;
import com.alibaba.polardbx.net.buffer.ByteBufferHolder;
import com.alibaba.polardbx.net.compress.IPacketOutputProxy;
import com.alibaba.polardbx.net.compress.PacketOutputProxyFactory;
import com.alibaba.polardbx.net.handler.QueryHandler;
import com.alibaba.polardbx.net.packet.*;
import com.alibaba.polardbx.net.util.CharsetUtil;
import com.alibaba.polardbx.rpc.pool.XConnection;
import com.alibaba.polardbx.rpc.result.XResult;
import com.alibaba.polardbx.rpc.result.XResultUtil;
import com.mysql.cj.polarx.protobuf.PolarxResultset;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.TimeZone;

/**
 * Demonstrates:
 * 1) Multi-server round-robin pooling via MultiServerConnectionPool
 * 2) Parallel chunk-splitting and merging
 * 3) Sending merged results back to a MySQL-protocol client
 * <p>
 * This is the main query handler that receives a SQL query from the client,
 * determines whether it can be split into parallel chunks, executes those
 * chunks, merges results, and sends them back.
 */
public class MyQueryHandler implements QueryHandler {

    private final ServerConnection frontendConnection;
    private final MultiServerConnectionPool connectionPool;
    private final ExecutorService executorService;

    /**
     * Constructs a query handler and initializes the connection pool with two servers.
     *
     * @param frontendConnection The front-end (client) connection that uses this handler.
     */
    public MyQueryHandler(ServerConnection frontendConnection) {
        this.frontendConnection = frontendConnection;

        // Create our custom multi-server pool and an executor service
        this.connectionPool = new MultiServerConnectionPool();
        this.executorService = Executors.newFixedThreadPool(2); // example: 2 worker threads

        try {
            // Example: Add two backend servers. Adjust as needed.
            ServerInfo mainServer = new ServerInfo(
                "10.1.1.148", 33660, "teste", "teste", "ssb"
            );
            connectionPool.addServer(mainServer);

            ServerInfo secondServer = new ServerInfo(
                "10.1.1.195", 33660, "teste", "teste", "ssb"
            );
            connectionPool.addServer(secondServer);

            System.out.println("Successfully initialized connection pool");
        } catch (Exception e) {
            throw new RuntimeException("Failed to initialize connection pool: " + e.getMessage());
        }
    }

    /**
     * Invoked when a query is received from the client. If the query can be chunk-split,
     * it runs in parallel. Otherwise, it does a single normal query.
     *
     * @param sql The SQL statement sent by the client.
     */
    @Override
    public void query(String sql) {
        System.out.println("Received query: " + sql);

        boolean chunkable = false;
        SQLStatement stmt = null;
        try {
            // Use Druid parser to check if it's a SELECT that we can chunk
            MySqlStatementParser parser = new MySqlStatementParser(sql);
            List<SQLStatement> list = parser.parseStatementList();
            if (!list.isEmpty() && list.get(0) instanceof SQLSelectStatement) {
                stmt = list.get(0);
                chunkable = canChunk((SQLSelectStatement) stmt);
            }
        } catch (ParserException pe) {
            System.err.println("Druid parse error: " + pe.getMessage());
        }

        if (chunkable && stmt instanceof SQLSelectStatement) {
            doParallelChunkQuery((SQLSelectStatement) stmt);
        } else {
            // Normal single-query path
            try {
                XConnection conn = connectionPool.getNextConnection();
                XResult result = conn.execQuery(sql);
                sendResultSetResponse(result);
            } catch (Exception e) {
                e.printStackTrace();
                sendErrorResponse("Query error: " + e.getMessage());
            }
        }
    }

    /**
     * Determines whether we can chunk-split the query. Here, we check if the table is "customer"
     * and there's an ORDER BY c_name.
     *
     * @param selectStmt The parsed SQLSelectStatement
     * @return true if chunk-splittable, false otherwise
     */
    private boolean canChunk(SQLSelectStatement selectStmt) {
        if (!(selectStmt.getSelect().getQuery() instanceof SQLSelectQueryBlock)) {
            return false;
        }
        SQLSelectQueryBlock qb = (SQLSelectQueryBlock) selectStmt.getSelect().getQuery();
        if (qb.getFrom() == null) {
            return false;
        }

        String tbl = qb.getFrom().toString().replace("`", "").trim().toLowerCase();
        if (!tbl.contains("customer")) {
            return false;
        }
        if (qb.getOrderBy() == null || qb.getOrderBy().getItems().size() != 1) {
            return false;
        }
        String orderCol = qb.getOrderBy().getItems().get(0).getExpr().toString().toLowerCase();
        return orderCol.contains("c_name");
    }

    /**
     * Performs a parallel chunked query:
     * 1) Builds range-based SQL queries.
     * 2) Submits each chunk query to an executor.
     * 3) Collects all chunk results.
     * 4) Merges them in ascending order by the first column.
     * 5) Sends a merged result set to the client.
     *
     * @param originalSelect The parsed SELECT statement.
     */
    private void doParallelChunkQuery(SQLSelectStatement originalSelect) {
        try {
            TimestampLogger.startTimer("fullQuery");

            // Example chunk definitions. These would normally be derived from data distribution.
            List<String> chunks = new ArrayList<>();
            chunks.add(buildChunkSQL(originalSelect, "< 3000000"));
            chunks.add(buildChunkSQL(originalSelect, ">= 3000000 AND c_custkey < 6000000"));

            List<Future<ChunkResult>> futures = new ArrayList<>();
            AtomicInteger idx = new AtomicInteger(0);

            // Submit each chunk to the executor pool
            for (String cSql : chunks) {
                futures.add(executorService.submit(() -> {
                    int chunkNo = idx.getAndIncrement();
                    String chunkId = "chunk" + chunkNo;

                    TimestampLogger.startTimer(chunkId);

                    // Acquire a connection and execute
                    XConnection conn = connectionPool.getNextConnection();
                    XResult res = conn.execQuery(cSql);

                    // Read all rows into memory for merging
                    List<List<Object>> rows = readAllRows(res);

                    TimestampLogger.logWithDuration(chunkId,
                        "Chunk " + chunkNo + " got " + rows.size() + " rows");

                    return new ChunkResult(res, rows);
                }));
            }

            // Collect results
            XResult metadataResult = null;
            List<List<List<Object>>> allChunks = new ArrayList<>();
            for (Future<ChunkResult> fut : futures) {
                ChunkResult cr = fut.get(300, TimeUnit.SECONDS);
                if (metadataResult == null) {
                    metadataResult = cr.result;
                }
                allChunks.add(cr.rows);
            }
            if (metadataResult == null) {
                throw new RuntimeException("No successful chunk to get metadata");
            }

            // Merge them all
            List<List<Object>> merged = mergeChunks(allChunks);

            // Send the final merged data to the client
            sendMergedResponse(metadataResult, merged);

            TimestampLogger.logWithDuration("fullQuery", "Parallel chunked query finished");
        } catch (Exception e) {
            e.printStackTrace();
            sendErrorResponse("Parallel-chunk error: " + e.getMessage());
        }
    }

    /**
     * Injects a range condition into the original SQL for chunking.
     *
     * @param original     The original parsed SQL statement
     * @param rangeCondition The condition (e.g., "c_custkey < 3000000")
     * @return The modified SQL string with the range condition inserted
     */
    private String buildChunkSQL(SQLSelectStatement original, String rangeCondition) {
        String originalSql = original.toString();
        int orderPos = originalSql.toLowerCase().indexOf("order by");
        if (orderPos < 0) {
            // No ORDER BY found; just append the condition
            return originalSql + " AND c_custkey " + rangeCondition;
        }

        // If there's an ORDER BY, insert the condition before it.
        String before = originalSql.substring(0, orderPos).trim();
        String after = originalSql.substring(orderPos);
        if (before.toLowerCase().contains("where")) {
            before += " AND c_custkey " + rangeCondition + " ";
        } else {
            before += " WHERE c_custkey " + rangeCondition + " ";
        }
        return before + after;
    }

    /**
     * Reads the entire XResult into a list of row objects. Each row is a List of columns.
     *
     * @param xres The XResult to read
     * @return A List of rows, each row is a List of Objects
     * @throws Exception If something goes wrong reading the result
     */
    private List<List<Object>> readAllRows(XResult xres) throws Exception {
        List<List<Object>> rows = new ArrayList<>();
        while (xres.next() != null) {
            List<Object> oneRow = new ArrayList<>();
            for (int i = 0; i < xres.getMetaData().size(); i++) {
                // Convert each column value to a Java Object
                Object val = XResultUtil.resultToObject(
                    xres.getMetaData().get(i),
                    xres.current().getRow().get(i),
                    true,
                    TimeZone.getDefault()
                ).getKey();
                oneRow.add(val);
            }
            rows.add(oneRow);
        }
        return rows;
    }

    /**
     * Merges multiple chunk-lists in ascending order by the first column.
     * This uses a PriorityQueue to do a multi-way merge.
     *
     * @param chunkRows A list of chunks, each chunk is a List of rows
     * @return A merged list of rows
     */
    private List<List<Object>> mergeChunks(List<List<List<Object>>> chunkRows) {
        // PriorityQueue that orders rows by the first column's numeric value
        PriorityQueue<ChunkIterator> pq = new PriorityQueue<>((a, b) -> {
            long vA = Long.parseLong(String.valueOf(a.current().get(0)));
            long vB = Long.parseLong(String.valueOf(b.current().get(0)));
            return Long.compare(vA, vB);
        });

        // Initialize the PQ with the first row from each chunk
        for (List<List<Object>> chunk : chunkRows) {
            if (!chunk.isEmpty()) {
                pq.add(new ChunkIterator(chunk));
            }
        }

        // Merge all rows in ascending order
        List<List<Object>> merged = new ArrayList<>();
        while (!pq.isEmpty()) {
            ChunkIterator top = pq.poll();
            merged.add(top.current());
            if (top.hasNext()) {
                top.next();
                pq.add(top);
            }
        }
        return merged;
    }

    /**
     * Sends a merged result set back to the client, using metadata from the first chunk's XResult.
     *
     * @param meta       An XResult that provides metadata about columns
     * @param mergedRows All merged rows
     */
    private void sendMergedResponse(XResult meta, List<List<Object>> mergedRows) {
        ByteBufferHolder buffer = null;
        try {
            byte packetId = 0;
            buffer = frontendConnection.allocate();
            buffer.clear();

            IPacketOutputProxy proxy = PacketOutputProxyFactory.getInstance().createProxy(frontendConnection, buffer);
            proxy.packetBegin();

            // 1) Result set header
            ResultSetHeaderPacket header = new ResultSetHeaderPacket();
            header.packetId = ++packetId;
            header.fieldCount = meta.getMetaData().size();
            header.write(proxy);

            // 2) Column definitions
            for (int i = 0; i < meta.getMetaData().size(); i++) {
                FieldPacket field = new FieldPacket();
                field.packetId = ++packetId;
                field.charsetIndex = CharsetUtil.getIndex("utf8");
                field.name = meta.getMetaData().get(i).getName().toByteArray();
                field.type = convertPolarDBTypeToMySQLType(meta.getMetaData().get(i));
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

            // 3) EOF (unless eofDeprecated is on)
            if (!frontendConnection.isEofDeprecated()) {
                EOFPacket eof = new EOFPacket();
                eof.packetId = ++packetId;
                eof.write(proxy);
            }

            // 4) Write each row
            for (List<Object> rowData : mergedRows) {
                RowDataPacket row = new RowDataPacket(meta.getMetaData().size());
                for (Object val : rowData) {
                    row.add(val != null ? val.toString().getBytes() : null);
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
                frontendConnection.recycleBuffer(buffer);
            }
        }
    }

    /**
     * Sends a normal (non-chunk) query result back to the client.
     *
     * @param result The XResult to send
     */
    private void sendResultSetResponse(XResult result) {
        ByteBufferHolder buffer = null;
        try {
            byte packetId = 0;
            buffer = frontendConnection.allocate();
            buffer.clear();

            IPacketOutputProxy proxy = PacketOutputProxyFactory.getInstance().createProxy(frontendConnection, buffer);
            proxy.packetBegin();

            // 1) Result set header
            ResultSetHeaderPacket header = new ResultSetHeaderPacket();
            header.packetId = ++packetId;
            header.fieldCount = result.getMetaData().size();
            header.write(proxy);

            // 2) Column definitions
            for (int i = 0; i < result.getMetaData().size(); i++) {
                FieldPacket field = new FieldPacket();
                field.packetId = ++packetId;
                field.charsetIndex = CharsetUtil.getIndex("utf8");
                field.name = result.getMetaData().get(i).getName().toByteArray();
                field.type = convertPolarDBTypeToMySQLType(result.getMetaData().get(i));
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

            // 3) EOF
            if (!frontendConnection.isEofDeprecated()) {
                EOFPacket eof = new EOFPacket();
                eof.packetId = ++packetId;
                eof.write(proxy);
            }

            // 4) Rows
            while (result.next() != null) {
                RowDataPacket row = new RowDataPacket(result.getMetaData().size());
                for (int i = 0; i < result.getMetaData().size(); i++) {
                    Object val = XResultUtil.resultToObject(
                        result.getMetaData().get(i),
                        result.current().getRow().get(i),
                        true,
                        TimeZone.getDefault()
                    ).getKey();
                    row.add(val != null ? val.toString().getBytes() : null);
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
                frontendConnection.recycleBuffer(buffer);
            }
        }
    }

    /**
     * Sends an error packet to the client.
     *
     * @param message The error message
     */
    private void sendErrorResponse(String message) {
        ByteBufferHolder buffer = null;
        try {
            buffer = frontendConnection.allocate();
            buffer.clear();

            IPacketOutputProxy proxy = PacketOutputProxyFactory.getInstance().createProxy(frontendConnection, buffer);
            proxy.packetBegin();

            ErrorPacket err = new ErrorPacket();
            err.packetId = 1;
            err.errno = 1064;
            err.message = message.getBytes();
            err.write(proxy);

            proxy.packetEnd();
        } catch (Exception e) {
            e.printStackTrace();
            if (buffer != null) {
                frontendConnection.recycleBuffer(buffer);
            }
        }
    }

    /**
     * Converts PolarxResultset ColumnMetaData to a MySQL type byte.
     * In this example, we treat everything as a VARCHAR for simplicity.
     */
    private byte convertPolarDBTypeToMySQLType(PolarxResultset.ColumnMetaData meta) {
        return (byte) Fields.FIELD_TYPE_VAR_STRING; // all as varstring
    }

    /**
     * Closes the executor service and the underlying connection pool.
     */
    public void close() {
        try {
            executorService.shutdown();
            if (!executorService.awaitTermination(10, TimeUnit.SECONDS)) {
                executorService.shutdownNow();
            }
        } catch (InterruptedException e) {
            executorService.shutdownNow();
        }
        connectionPool.close();
    }
}
