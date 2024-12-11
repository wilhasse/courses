import com.alibaba.polardbx.Fields;
import com.alibaba.polardbx.net.FrontendConnection;
import com.alibaba.polardbx.net.NIOAcceptor;
import com.alibaba.polardbx.net.NIOProcessor;
import com.alibaba.polardbx.net.buffer.ByteBufferHolder;
import com.alibaba.polardbx.net.compress.IPacketOutputProxy;
import com.alibaba.polardbx.net.compress.PacketOutputProxyFactory;
import com.alibaba.polardbx.net.factory.FrontendConnectionFactory;
import com.alibaba.polardbx.net.handler.QueryHandler;
import com.alibaba.polardbx.net.packet.EOFPacket;
import com.alibaba.polardbx.net.packet.FieldPacket;
import com.alibaba.polardbx.net.packet.ResultSetHeaderPacket;
import com.alibaba.polardbx.net.packet.RowDataPacket;
import com.alibaba.polardbx.net.util.CharsetUtil;
import com.alibaba.polardbx.net.util.TimeUtil;
import com.alibaba.polardbx.common.utils.thread.ThreadCpuStatUtil;

import java.io.IOException;
import java.nio.channels.SocketChannel;
import java.util.Timer;
import java.util.TimerTask;

public class SimpleServer {
    private static final int SERVER_PORT = 8507;
    private NIOProcessor[] processors;
    private NIOAcceptor server;

    public static void main(String[] args) throws IOException {
        SimpleServer server = new SimpleServer();
        server.startup();
    }

    public void startup() throws IOException {
        // Initialize timer for time-based operations
        Timer timer = new Timer("ServerTimer", true);
        timer.schedule(new TimerTask() {
            @Override
            public void run() {
                TimeUtil.update();
            }
        }, 0L, 100L);

        // Create processors based on CPU cores
        processors = new NIOProcessor[ThreadCpuStatUtil.NUM_CORES];
        for (int i = 0; i < processors.length; i++) {
            processors[i] = new NIOProcessor(i, "Processor" + i, 4);
            processors[i].startup();
        }

        // Create and start server
        CustomConnectionFactory factory = new CustomConnectionFactory();
        server = new NIOAcceptor("MySQLServer", SERVER_PORT, factory, true);
        server.setProcessors(processors);
        server.start();

        System.out.println("MySQL server started on port " + SERVER_PORT);
    }
}

class CustomConnectionFactory extends FrontendConnectionFactory {
    @Override
    protected FrontendConnection getConnection(SocketChannel channel) {
        CustomConnection conn = new CustomConnection(channel);
        conn.setQueryHandler(new VersionQueryHandler(conn));
        return conn;
    }
}

class CustomConnection extends FrontendConnection {
    public CustomConnection(SocketChannel channel) {
        super(channel);
    }

    @Override
    public void handleError(com.alibaba.polardbx.common.exception.code.ErrorCode errorCode, Throwable t) {
        System.err.println("Error: " + t.getMessage());
        close();
    }

    @Override
    public boolean checkConnectionCount() {
        return true;
    }

    @Override
    protected long genConnId() {
        return 1; // Simplified for example
    }

    @Override
    public boolean isPrivilegeMode() {
        return false;  // Added missing method
    }

    @Override
    public void addConnectionCount() {
        // No-op for simple implementation
    }

    @Override
    public boolean prepareLoadInfile(String sql) {
        return false;  // We don't support LOAD INFILE
    }

    @Override
    public void binlogDump(byte[] data) {
        // No-op for simple implementation
    }

    @Override
    public void fieldList(byte[] data) {
        // No-op for simple implementation
    }
}

class VersionQueryHandler implements QueryHandler {
    private final CustomConnection connection;

    public VersionQueryHandler(CustomConnection connection) {
        this.connection = connection;
    }

    @Override
    public void query(String sql) {
        if (sql.toLowerCase().contains("select version()")) {
            sendVersionResponse();
        } else {
            // For any other query, send empty result
            sendEmptyResponse();
        }
    }

    private void sendVersionResponse() {
        byte packetId = 0;
        ByteBufferHolder buffer = connection.allocate();
        IPacketOutputProxy proxy = PacketOutputProxyFactory.getInstance().createProxy(connection, buffer);

        // Header - 1 column
        ResultSetHeaderPacket header = new ResultSetHeaderPacket();
        header.packetId = ++packetId;
        header.fieldCount = 1;
        header.write(proxy);

        // Field packet
        FieldPacket field = new FieldPacket();
        field.packetId = ++packetId;
        field.charsetIndex = CharsetUtil.getIndex("utf8");
        field.name = "VERSION()".getBytes();
        field.type = Fields.FIELD_TYPE_VAR_STRING;
        field.write(proxy);

        // EOF packet after fields
        if (!connection.isEofDeprecated()) {
            EOFPacket eof = new EOFPacket();
            eof.packetId = ++packetId;
            eof.write(proxy);
        }

        // Row data
        RowDataPacket row = new RowDataPacket(1);
        row.add("PolarDB-X 5.4.19-SNAPSHOT".getBytes());
        row.packetId = ++packetId;
        row.write(proxy);

        // EOF packet after rows
        EOFPacket lastEof = new EOFPacket();
        lastEof.packetId = ++packetId;
        lastEof.write(proxy);

        // Send the response
        connection.write(buffer);
    }

    private void sendEmptyResponse() {
        byte packetId = 0;
        ByteBufferHolder buffer = connection.allocate();
        IPacketOutputProxy proxy = PacketOutputProxyFactory.getInstance().createProxy(connection, buffer);

        ResultSetHeaderPacket header = new ResultSetHeaderPacket();
        header.packetId = ++packetId;
        header.fieldCount = 0;
        header.write(proxy);

        EOFPacket lastEof = new EOFPacket();
        lastEof.packetId = ++packetId;
        lastEof.write(proxy);

        connection.write(buffer);
    }
}