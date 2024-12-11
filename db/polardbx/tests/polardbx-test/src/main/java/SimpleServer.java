import com.alibaba.polardbx.Fields;
import com.alibaba.polardbx.net.FrontendConnection;
import com.alibaba.polardbx.net.NIOAcceptor;
import com.alibaba.polardbx.net.NIOProcessor;
import com.alibaba.polardbx.net.buffer.BufferPool;
import com.alibaba.polardbx.net.buffer.ByteBufferHolder;
import com.alibaba.polardbx.net.compress.IPacketOutputProxy;
import com.alibaba.polardbx.net.compress.PacketOutputProxyFactory;
import com.alibaba.polardbx.net.factory.FrontendConnectionFactory;
import com.alibaba.polardbx.net.handler.QueryHandler;
import com.alibaba.polardbx.net.handler.Privileges;
import com.alibaba.polardbx.net.packet.EOFPacket;
import com.alibaba.polardbx.net.packet.FieldPacket;
import com.alibaba.polardbx.net.packet.ResultSetHeaderPacket;
import com.alibaba.polardbx.net.packet.RowDataPacket;
import com.alibaba.polardbx.net.util.CharsetUtil;
import com.alibaba.polardbx.net.util.TimeUtil;
import com.alibaba.polardbx.common.utils.thread.ThreadCpuStatUtil;
import com.alibaba.polardbx.common.utils.thread.ServerThreadPool;
import com.alibaba.polardbx.common.utils.logger.Logger;
import com.alibaba.polardbx.common.utils.logger.LoggerFactory;
import com.taobao.tddl.common.privilege.EncrptPassword;

import java.util.concurrent.atomic.AtomicLong;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.SocketChannel;
import java.util.Timer;
import java.util.TimerTask;
import java.util.Map;
import java.util.HashMap;
import java.util.Set;

public class SimpleServer {
    private static final int SERVER_PORT = 8507;
    private static final SimpleServer INSTANCE = new SimpleServer();
    private SimpleConfig config;
    private NIOProcessor[] processors;
    private NIOAcceptor server;

    public static SimpleServer getInstance() {
        return INSTANCE;
    }

    private SimpleServer() {
        this.config = new SimpleConfig();
    }

    public SimpleConfig getConfig() {
        return config;
    }

    public void startup() throws IOException {
        System.out.println("Starting server initialization...");

        // Initialize timer for time-based operations
        Timer timer = new Timer("ServerTimer", true);
        timer.schedule(new TimerTask() {
            @Override
            public void run() {
                TimeUtil.update();
            }
        }, 0L, 100L);
        System.out.println("Timer initialized");

        // Create processors based on CPU cores
        int processorCount = Math.max(1, ThreadCpuStatUtil.NUM_CORES);
        System.out.println("Creating " + processorCount + " processors");

        processors = new NIOProcessor[processorCount];
        for (int i = 0; i < processors.length; i++) {
            // Create a ServerThreadPool with 1 bucket to avoid divide by zero
            ServerThreadPool handler = new ServerThreadPool(
                    "ProcessorHandler-" + i,
                    4,  // poolSize
                    5000,  // deadLockCheckPeriod (5 seconds)
                    1   // bucketSize
            );

            processors[i] = new NIOProcessor(i, "Processor" + i, handler);
            processors[i].startup();
            System.out.println("Processor " + i + " started");
        }

        // Create and start server
        DebugConnectionFactory factory = new DebugConnectionFactory();
        server = new NIOAcceptor("MySQLServer", SERVER_PORT, factory, true);
        server.setProcessors(processors);
        server.start();

        System.out.println("Server started on port " + SERVER_PORT);
    }

    static class SimpleConfig {
        private final Map<String, String> users;

        public SimpleConfig() {
            this.users = new HashMap<>();
            this.users.put("root", "12345");
        }

        public Map<String, String> getUsers() {
            return users;
        }
    }

    class DebugConnection extends FrontendConnection {
        private final Logger logger = LoggerFactory.getLogger(DebugConnection.class);
        private final BufferPool bufferPool;
        private final AtomicLong CONNECTION_ID = new AtomicLong(1);
        private final long connectionId;

        public DebugConnection(SocketChannel channel) {
            super(channel);
            this.bufferPool = new BufferPool(1024 * 1024 * 16, 4096);
            this.packetHeaderSize = 4;  // MySQL packet header size
            this.maxPacketSize = 16 * 1024 * 1024;  // 16MB max packet
            this.readBuffer = allocate();  // Initialize read buffer
            this.connectionId = CONNECTION_ID.getAndIncrement();
            System.out.println("Created new connection " + connectionId + " with buffer pool");
        }

        @Override
        public ByteBufferHolder allocate() {
            try {
                ByteBufferHolder buffer = bufferPool.allocate();
                if (buffer != null) {
                    ByteBuffer nioBuffer = buffer.getBuffer();
                    if (nioBuffer != null) {
                        nioBuffer.clear();
                        nioBuffer.position(0);
                        nioBuffer.limit(nioBuffer.capacity());
                    }
                }
                return buffer != null ? buffer : ByteBufferHolder.EMPTY;
            } catch (Exception e) {
                logger.error("Error allocating buffer", e);
                return ByteBufferHolder.EMPTY;
            }
        }

        public void recycleBuffer(ByteBufferHolder buffer) {
            if (buffer != null && buffer != ByteBufferHolder.EMPTY) {
                try {
                    bufferPool.recycle(buffer);
                } catch (Exception e) {
                    logger.error("Error recycling buffer", e);
                }
            }
        }

        @Override
        public void handleError(com.alibaba.polardbx.common.exception.code.ErrorCode errorCode, Throwable t) {
            logger.error("Connection error - Code: " + errorCode, t);
            close();
        }

        @Override
        public boolean checkConnectionCount() {
            return true;
        }

        @Override
        public void addConnectionCount() {
        }

        @Override
        public boolean isPrivilegeMode() {
            return true;
        }

        @Override
        protected long genConnId() {
            return connectionId;
        }

        @Override
        public long getId() {
            return connectionId;
        }

        @Override
        public boolean prepareLoadInfile(String sql) {
            return false;
        }

        @Override
        public void binlogDump(byte[] data) {
        }

        @Override
        public void fieldList(byte[] data) {
        }
    }

    class DebugConnectionFactory extends FrontendConnectionFactory {
        @Override
        protected FrontendConnection getConnection(SocketChannel channel) {
            System.out.println("Creating new connection for channel: " + channel);
            DebugConnection c = new DebugConnection(channel);
            c.setPrivileges(new SimplePrivileges());
            c.setQueryHandler(new SimpleQueryHandler(c));
            return c;
        }
    }

    class SimplePrivileges implements Privileges {
        @Override
        public boolean schemaExists(String schema) {
            return true;
        }

        @Override
        public boolean userExists(String user) {
            return getConfig().getUsers().containsKey(user);
        }

        @Override
        public boolean userExists(String user, String host) {
            return userExists(user);
        }

        @Override
        public boolean userMatches(String user, String host) {
            return true;
        }

        @Override
        public EncrptPassword getPassword(String user) {
            String pass = getConfig().getUsers().get(user);
            return new EncrptPassword(pass, false);
        }

        @Override
        public EncrptPassword getPassword(String user, String host) {
            return getPassword(user);
        }

        @Override
        public Set<String> getUserSchemas(String user) {
            return null;
        }

        @Override
        public Set<String> getUserSchemas(String user, String host) {
            return null;
        }

        @Override
        public boolean isTrustedIp(String host, String user) {
            return true;
        }

        @Override
        public Map<String, com.alibaba.polardbx.common.model.DbPriv> getSchemaPrivs(String user, String host) {
            return null;
        }

        @Override
        public Map<String, com.alibaba.polardbx.common.model.TbPriv> getTablePrivs(String user, String host, String database) {
            return null;
        }

        @Override
        public boolean checkQuarantine(String user, String host) {
            return true;
        }
    }

    class SimpleQueryHandler implements QueryHandler {
        private final DebugConnection connection;

        public SimpleQueryHandler(DebugConnection connection) {
            this.connection = connection;
            System.out.println("Created query handler for connection: " + connection);
        }

        @Override
        public void query(String sql) {
            System.out.println("Received query: " + sql);
            String sqlLower = sql.toLowerCase();
            if (sqlLower.contains("select version()")) {
                System.out.println("Processing VERSION query");
                sendVersionResponse();
            } else if (sqlLower.contains("connection_id()")) {
                System.out.println("Processing CONNECTION_ID query");
                sendConnectionIdResponse();
            } else {
                System.out.println("Processing unknown query");
                sendEmptyResponse();
            }
        }

        private void sendVersionResponse() {
            ByteBufferHolder buffer = null;
            try {
                // Initialize buffer
                byte packetId = 0;
                buffer = connection.allocate();
                buffer.clear();  // Reset buffer positions
                System.out.println("Allocated buffer for version response");

                // Create proxy
                IPacketOutputProxy proxy = PacketOutputProxyFactory.getInstance().createProxy(connection, buffer);
                proxy.packetBegin();

                // Write header
                System.out.println("Writing header packet");
                ResultSetHeaderPacket header = new ResultSetHeaderPacket();
                header.packetId = ++packetId;
                header.fieldCount = 1;
                header.write(proxy);

                // Write field
                System.out.println("Writing field packet");
                FieldPacket field = new FieldPacket();
                field.packetId = ++packetId;
                field.charsetIndex = CharsetUtil.getIndex("utf8");
                field.name = "VERSION()".getBytes();
                field.type = Fields.FIELD_TYPE_VAR_STRING;
                field.catalog = "def".getBytes();
                field.db = new byte[0];
                field.table = new byte[0];
                field.orgTable = new byte[0];
                field.orgName = field.name;
                field.decimals = 0;
                field.flags = 0;
                field.length = 50;
                field.write(proxy);

                // Write EOF if needed
                if (!connection.isEofDeprecated()) {
                    System.out.println("Writing first EOF packet");
                    EOFPacket eof = new EOFPacket();
                    eof.packetId = ++packetId;
                    eof.write(proxy);
                }

                // Write row
                System.out.println("Writing row packet");
                RowDataPacket row = new RowDataPacket(1);
                row.add("PolarDB-X 5.4.19-SNAPSHOT".getBytes());
                row.packetId = ++packetId;
                row.write(proxy);

                // Write final EOF
                System.out.println("Writing final EOF packet");
                EOFPacket lastEof = new EOFPacket();
                lastEof.packetId = ++packetId;
                lastEof.write(proxy);

                // Finish packet
                proxy.packetEnd();

            } catch (Exception e) {
                System.err.println("Error in sendVersionResponse: " + e);
                e.printStackTrace();
                if (buffer != null) {
                    connection.recycleBuffer(buffer);
                }
            }
        }

        private void sendConnectionIdResponse() {
            ByteBufferHolder buffer = null;
            try {
                // Initialize buffer
                byte packetId = 0;
                buffer = connection.allocate();
                buffer.clear();  // Reset buffer positions
                System.out.println("Allocated buffer for connection_id response");

                // Create proxy
                IPacketOutputProxy proxy = PacketOutputProxyFactory.getInstance().createProxy(connection, buffer);
                proxy.packetBegin();

                // Write header
                System.out.println("Writing header packet");
                ResultSetHeaderPacket header = new ResultSetHeaderPacket();
                header.packetId = ++packetId;
                header.fieldCount = 1;
                header.write(proxy);

                // Write field
                System.out.println("Writing field packet");
                FieldPacket field = new FieldPacket();
                field.packetId = ++packetId;
                field.charsetIndex = CharsetUtil.getIndex("utf8");
                field.name = "CONNECTION_ID()".getBytes();
                field.type = Fields.FIELD_TYPE_LONGLONG;  // 64-bit integer
                field.catalog = "def".getBytes();
                field.db = new byte[0];
                field.table = new byte[0];
                field.orgTable = new byte[0];
                field.orgName = field.name;
                field.decimals = 0;
                field.flags = 0;
                field.length = 20;  // Long enough for connection ID
                field.write(proxy);

                // Write EOF if needed
                if (!connection.isEofDeprecated()) {
                    System.out.println("Writing first EOF packet");
                    EOFPacket eof = new EOFPacket();
                    eof.packetId = ++packetId;
                    eof.write(proxy);
                }

                // Write row with the connection ID
                System.out.println("Writing row packet");
                RowDataPacket row = new RowDataPacket(1);
                row.add(String.valueOf(connection.getId()).getBytes());
                row.packetId = ++packetId;
                row.write(proxy);

                // Write final EOF
                System.out.println("Writing final EOF packet");
                EOFPacket lastEof = new EOFPacket();
                lastEof.packetId = ++packetId;
                lastEof.write(proxy);

                // Finish packet
                proxy.packetEnd();

            } catch (Exception e) {
                System.err.println("Error in sendConnectionIdResponse: " + e);
                e.printStackTrace();
                if (buffer != null) {
                    connection.recycleBuffer(buffer);
                }
            }
        }

        private void sendEmptyResponse() {
            ByteBufferHolder buffer = null;
            try {
                byte packetId = 0;
                buffer = connection.allocate();
                buffer.clear();  // Reset buffer positions

                IPacketOutputProxy proxy = PacketOutputProxyFactory.getInstance().createProxy(connection, buffer);
                proxy.packetBegin();

                ResultSetHeaderPacket header = new ResultSetHeaderPacket();
                header.packetId = ++packetId;
                header.fieldCount = 0;
                header.write(proxy);

                EOFPacket lastEof = new EOFPacket();
                lastEof.packetId = ++packetId;
                lastEof.write(proxy);

                proxy.packetEnd();
            } catch (Exception e) {
                System.err.println("Error sending empty response: " + e);
                e.printStackTrace();
                if (buffer != null) {
                    connection.recycleBuffer(buffer);
                }
            }
        }
    }

    public static void main(String[] args) {
        try {
            getInstance().startup();

            System.out.println("Server started successfully, press Ctrl+C to stop");
            while (true) {
                Thread.sleep(1000);
            }
        } catch (Exception e) {
            System.err.println("Server failed to start: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}