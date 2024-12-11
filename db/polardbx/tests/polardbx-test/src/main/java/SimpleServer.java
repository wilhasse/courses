import com.alibaba.polardbx.Fields;
import com.alibaba.polardbx.net.FrontendConnection;
import com.alibaba.polardbx.net.NIOAcceptor;
import com.alibaba.polardbx.net.NIOProcessor;
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
import com.taobao.tddl.common.privilege.EncrptPassword;

import java.io.IOException;
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
            processors[i] = new NIOProcessor(i, "Processor" + i, 4);
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
        public DebugConnection(SocketChannel channel) {
            super(channel);
            System.out.println("New connection created from: " + channel);
        }

        @Override
        public void handleError(com.alibaba.polardbx.common.exception.code.ErrorCode errorCode, Throwable t) {
            System.err.println("Connection error - Code: " + errorCode + ", Message: " + t.getMessage());
            t.printStackTrace();
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
            return 1;
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

        // Override other methods to add logging
        @Override
        public void read() throws IOException {
            System.out.println("Reading from connection: " + this);
            super.read();
        }

        @Override
        public void write(ByteBufferHolder buffer) {
            System.out.println("Writing to connection: " + this);
            super.write(buffer);
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
            if (sql.toLowerCase().contains("select version()")) {
                System.out.println("Processing VERSION query");
                sendVersionResponse();
            } else {
                System.out.println("Processing unknown query");
                sendEmptyResponse();
            }
        }

        private void sendVersionResponse() {
            try {
                byte packetId = 0;
                ByteBufferHolder buffer = connection.allocate();
                IPacketOutputProxy proxy = PacketOutputProxyFactory.getInstance().createProxy(connection, buffer);

                System.out.println("Sending version response header");
                ResultSetHeaderPacket header = new ResultSetHeaderPacket();
                header.packetId = ++packetId;
                header.fieldCount = 1;
                header.write(proxy);

                System.out.println("Sending field packet");
                FieldPacket field = new FieldPacket();
                field.packetId = ++packetId;
                field.charsetIndex = CharsetUtil.getIndex("utf8");
                field.name = "VERSION()".getBytes();
                field.type = Fields.FIELD_TYPE_VAR_STRING;
                field.write(proxy);

                if (!connection.isEofDeprecated()) {
                    System.out.println("Sending EOF packet");
                    EOFPacket eof = new EOFPacket();
                    eof.packetId = ++packetId;
                    eof.write(proxy);
                }

                System.out.println("Sending row data");
                RowDataPacket row = new RowDataPacket(1);
                row.add("PolarDB-X 5.4.19-SNAPSHOT".getBytes());
                row.packetId = ++packetId;
                row.write(proxy);

                System.out.println("Sending final EOF packet");
                EOFPacket lastEof = new EOFPacket();
                lastEof.packetId = ++packetId;
                lastEof.write(proxy);

                System.out.println("Writing buffer to connection");
                connection.write(buffer);
            } catch (Exception e) {
                System.err.println("Error sending version response: " + e.getMessage());
                e.printStackTrace();
            }
        }

        private void sendEmptyResponse() {
            System.out.println("Sending empty response");
            try {
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
            } catch (Exception e) {
                System.err.println("Error sending empty response: " + e.getMessage());
                e.printStackTrace();
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