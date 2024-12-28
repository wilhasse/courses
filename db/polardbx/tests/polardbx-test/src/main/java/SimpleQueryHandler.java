import com.alibaba.polardbx.Fields;
import com.alibaba.polardbx.net.buffer.ByteBufferHolder;
import com.alibaba.polardbx.net.compress.IPacketOutputProxy;
import com.alibaba.polardbx.net.compress.PacketOutputProxyFactory;
import com.alibaba.polardbx.net.handler.QueryHandler;
import com.alibaba.polardbx.net.packet.*;
import com.alibaba.polardbx.net.util.CharsetUtil;
import com.alibaba.polardbx.rpc.pool.XConnection;
import com.alibaba.polardbx.rpc.pool.XConnectionManager;
import com.alibaba.polardbx.rpc.result.XResult;
import com.alibaba.polardbx.rpc.result.XResultUtil;
import com.mysql.cj.polarx.protobuf.PolarxResultset;

import java.util.TimeZone;

public class SimpleQueryHandler implements QueryHandler {
    protected final DebugConnection connection;
    protected final XConnection polardbConnection;
    protected final XConnectionManager manager;

    public SimpleQueryHandler(DebugConnection connection) {
        this.connection = connection;
        this.manager = XConnectionManager.getInstance();
        System.out.println("Created query handler for connection: " + connection);

        try {
            String host = "10.1.1.148";
            int port = 33660;
            String username = "teste";
            String password = "teste";
            String defaultDB = "ssb";
            long timeoutNanos = 30000 * 1000000L;

            manager.initializeDataSource(host, port, username, password, "test-instance");
            this.polardbConnection = manager.getConnection(host, port, username, password, defaultDB, timeoutNanos);
            this.polardbConnection.setStreamMode(true);
            this.polardbConnection.execUpdate("USE " + defaultDB);

            System.out.println("Connected to PolarDB-X engine");
        } catch (Exception e) {
            throw new RuntimeException("Failed to connect to PolarDB-X: " + e.getMessage(), e);
        }
    }

    public void close() {
        try {
            if (polardbConnection != null) {
                polardbConnection.close();
            }
            manager.deinitializeDataSource("10.1.1.148", 33660, "teste", "teste");
        } catch (Exception e) {
            System.err.println("Error closing PolarDB-X connection: " + e);
        }
    }

    @Override
    public void query(String sql) {
        System.out.println("Received query: " + sql);
        try {
            XResult result = polardbConnection.execQuery(sql);
            sendResultSetResponse(result);
        } catch (Exception e) {
            System.err.println("Error executing query on PolarDB-X: " + e.getMessage());
            e.printStackTrace();
            sendErrorResponse(e.getMessage());
        }
    }

    protected void sendResultSetResponse(XResult result) {
        ByteBufferHolder buffer = null;
        try {
            byte packetId = 0;
            buffer = connection.allocate();
            buffer.clear();

            IPacketOutputProxy proxy = PacketOutputProxyFactory.getInstance().createProxy(connection, buffer);
            proxy.packetBegin();

            // Write header
            ResultSetHeaderPacket header = new ResultSetHeaderPacket();
            header.packetId = ++packetId;
            header.fieldCount = result.getMetaData().size();
            header.write(proxy);

            // Write fields
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

            // Write EOF
            if (!connection.isEofDeprecated()) {
                EOFPacket eof = new EOFPacket();
                eof.packetId = ++packetId;
                eof.write(proxy);
            }

            // Write rows
            while (result.next() != null) {
                RowDataPacket row = new RowDataPacket(result.getMetaData().size());
                for (int i = 0; i < result.getMetaData().size(); i++) {
                    Object value = XResultUtil.resultToObject(
                            result.getMetaData().get(i),
                            result.current().getRow().get(i),
                            true,
                            TimeZone.getDefault()
                    ).getKey();

                    row.add(value != null ? value.toString().getBytes() : null);
                }
                row.packetId = ++packetId;
                row.write(proxy);
            }

            // Write final EOF
            EOFPacket lastEof = new EOFPacket();
            lastEof.packetId = ++packetId;
            lastEof.write(proxy);

            proxy.packetEnd();
        } catch (Exception e) {
            System.err.println("Error sending result set: " + e);
            e.printStackTrace();
            if (buffer != null) {
                connection.recycleBuffer(buffer);
            }
        }
    }

    protected void sendErrorResponse(String message) {
        ByteBufferHolder buffer = null;
        try {
            buffer = connection.allocate();
            buffer.clear();

            IPacketOutputProxy proxy = PacketOutputProxyFactory.getInstance().createProxy(connection, buffer);
            proxy.packetBegin();

            ErrorPacket err = new ErrorPacket();
            err.packetId = (byte)1;
            err.errno = (short)1064;
            err.message = message.getBytes();
            err.write(proxy);

            proxy.packetEnd();
        } catch (Exception e) {
            System.err.println("Error sending error response: " + e);
            e.printStackTrace();
            if (buffer != null) {
                connection.recycleBuffer(buffer);
            }
        }
    }

    protected byte convertPolarDBTypeToMySQLType(PolarxResultset.ColumnMetaData metaData) {
        // Cast to byte will preserve the correct bits for MySQL protocol
        // 253 as int -> 11111101 in binary -> -3 as signed byte
        // When transmitted, it will be read correctly as 253 by MySQL clients
        return (byte) Fields.FIELD_TYPE_VAR_STRING;          }
}
