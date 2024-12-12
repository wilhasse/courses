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
import com.mysql.cj.polarx.protobuf.PolarxResultset.ColumnMetaData;
import com.alibaba.polardbx.net.packet.ErrorPacket;

// PolarDBX Connection
import com.alibaba.polardbx.rpc.pool.XConnection;
import com.alibaba.polardbx.rpc.pool.XConnectionManager;
import com.alibaba.polardbx.rpc.result.XResult;
import com.alibaba.polardbx.rpc.result.XResultUtil;

import java.util.TimeZone;
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
        private final SimpleQueryHandler queryHandler;

        public DebugConnection(SocketChannel channel) {
            super(channel);
            this.bufferPool = new BufferPool(1024 * 1024 * 16, 4096);
            this.packetHeaderSize = 4;
            this.maxPacketSize = 16 * 1024 * 1024;
            this.readBuffer = allocate();
            this.queryHandler = new SimpleQueryHandler(this);
            this.connectionId = CONNECTION_ID.getAndIncrement();
            System.out.println("Created new connection " + connectionId + " with buffer pool");
            System.out.println("Created new connection with buffer pool");
        }

        @Override
        protected void cleanup() {
            try {
                //queryHandler.closeConnection();
            } catch (Exception e) {
                logger.error("Error closing PolarDB-X connection", e);
            }
            super.cleanup();
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
        private final XConnection polardbConnection;
        private final XConnectionManager manager;

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

        private void sendResultSetResponse(XResult result) {
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

        private void sendErrorResponse(String message) {
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

        private byte convertPolarDBTypeToMySQLType(ColumnMetaData metaData) {
            // Cast to byte will preserve the correct bits for MySQL protocol
            // 253 as int -> 11111101 in binary -> -3 as signed byte
            // When transmitted, it will be read correctly as 253 by MySQL clients
            return (byte)Fields.FIELD_TYPE_VAR_STRING;          }
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