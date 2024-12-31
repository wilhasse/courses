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

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

public class SimpleQueryHandler implements QueryHandler {
    protected final DebugConnection connection;
    protected final MultiServerConnectionPool connectionPool;
    protected final XConnectionManager manager;
    private static final int POOL_SIZE_PER_SERVER = 1;

    protected class ServerInfo {
        final String host;
        final int port;
        final String username;
        final String password;
        final String defaultDB;

        public ServerInfo(String host, int port, String username, String password, String defaultDB) {
            this.host = host;
            this.port = port;
            this.username = username;
            this.password = password;
            this.defaultDB = defaultDB;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            ServerInfo that = (ServerInfo) o;
            return port == that.port &&
                    Objects.equals(host, that.host) &&
                    Objects.equals(username, that.username);
        }

        @Override
        public int hashCode() {
            return Objects.hash(host, port, username);
        }
    }

    protected class MultiServerConnectionPool {
        private final Map<ServerInfo, List<XConnection>> serverConnections;
        private final List<ServerInfo> serverList;
        private final AtomicInteger nextServerIndex = new AtomicInteger(0);
        private final AtomicInteger nextConnectionIndex = new AtomicInteger(0);
        private final Object connectionLock = new Object();

        public MultiServerConnectionPool() {
            this.serverConnections = new ConcurrentHashMap<>();
            this.serverList = Collections.synchronizedList(new ArrayList<>());
        }

        public XConnection getNextConnection() {
            synchronized(connectionLock) {
                if (serverList.isEmpty()) {
                    throw new IllegalStateException("No servers available in the pool");
                }

                // Round-robin between servers
                int currentServerIndex = nextServerIndex.getAndIncrement() % serverList.size();
                ServerInfo server = serverList.get(currentServerIndex);

                List<XConnection> connections = serverConnections.get(server);
                if (connections == null || connections.isEmpty()) {
                    throw new IllegalStateException("No connections available for server " +
                            server.host + ":" + server.port);
                }

                // Round-robin between connections
                int connIndex = nextConnectionIndex.getAndIncrement() % connections.size();
                XConnection conn = connections.get(connIndex);

                // Create new connection if current one is invalid
                try {
                    if (!isConnectionValid(conn)) {
                        conn = createConnection(server);
                        connections.set(connIndex, conn);
                    }
                } catch (Exception e) {
                    System.err.println("Error with connection to " + server.host + ": " + e.getMessage());
                    // Try to create a new connection
                    try {
                        conn = createConnection(server);
                        connections.set(connIndex, conn);
                    } catch (Exception e2) {
                        throw new RuntimeException("Failed to create new connection: " + e2.getMessage());
                    }
                }

                return conn;
            }
        }

        private boolean isConnectionValid(XConnection conn) {
            try {
                XResult result = conn.execQuery("SELECT 1");
                // Make sure to finish fetching all results
                while (result.next() != null) {
                    // Consume all results
                }
                return true;
            } catch (Exception e) {
                return false;
            }
        }

        private XConnection createConnection(ServerInfo serverInfo) throws Exception {
            synchronized(connectionLock) {
                XConnection conn = manager.getConnection(
                        serverInfo.host, serverInfo.port,
                        serverInfo.username, serverInfo.password,
                        serverInfo.defaultDB, 30000 * 1000000L
                );
                conn.setStreamMode(true);

                // Execute USE statement and consume all results
                XResult result = conn.execQuery("USE " + serverInfo.defaultDB);
                while (result.next() != null) {
                    // Consume all results
                }

                return conn;
            }
        }

        public void addServer(ServerInfo serverInfo) {
            synchronized(connectionLock) {
                if (!serverConnections.containsKey(serverInfo)) {
                    try {
                        manager.initializeDataSource(serverInfo.host, serverInfo.port,
                                serverInfo.username, serverInfo.password, "test-instance");

                        List<XConnection> connections = new ArrayList<>(POOL_SIZE_PER_SERVER);
                        for (int i = 0; i < POOL_SIZE_PER_SERVER; i++) {
                            try {
                                XConnection conn = createConnection(serverInfo);
                                if (conn != null) {
                                    connections.add(conn);
                                }
                            } catch (Exception e) {
                                System.err.println("Failed to create connection " + i +
                                        " for server " + serverInfo.host + ": " + e.getMessage());
                            }
                        }

                        if (!connections.isEmpty()) {
                            serverConnections.put(serverInfo, connections);
                            serverList.add(serverInfo);
                            System.out.println("Added server " + serverInfo.host +
                                    " with " + connections.size() + " connections");
                        }
                    } catch (Exception e) {
                        System.err.println("Failed to initialize server " +
                                serverInfo.host + ": " + e.getMessage());
                    }
                }
            }
        }

        public void close() {
            synchronized(connectionLock) {
                for (Map.Entry<ServerInfo, List<XConnection>> entry : serverConnections.entrySet()) {
                    ServerInfo server = entry.getKey();
                    List<XConnection> connections = entry.getValue();

                    for (XConnection conn : connections) {
                        try {
                            if (conn != null) {
                                conn.close();
                            }
                        } catch (Exception e) {
                            System.err.println("Error closing connection to " +
                                    server.host + ": " + e);
                        }
                    }

                    manager.deinitializeDataSource(server.host, server.port,
                            server.username, server.password);
                }
            }
        }
    }

    public SimpleQueryHandler(DebugConnection connection) {
        this.connection = connection;
        this.manager = XConnectionManager.getInstance();
        this.connectionPool = new MultiServerConnectionPool();

        try {
            // Add servers with error handling
            ServerInfo mainServer = new ServerInfo(
                    "10.1.1.156", 33660, "teste", "teste", "ssb"
            );
            connectionPool.addServer(mainServer);

            ServerInfo secondServer = new ServerInfo(
                    "10.1.1.157", 33660, "teste", "teste", "ssb"
            );
            connectionPool.addServer(secondServer);

            System.out.println("Successfully initialized connection pool");
        } catch (Exception e) {
            throw new RuntimeException("Failed to initialize connection pool: " + e.getMessage());
        }
    }

    @Override
    public void query(String sql) {
        System.out.println("Received query: " + sql);
        try {
            XResult result = connectionPool.getNextConnection().execQuery(sql);
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
            err.packetId = (byte) 1;
            err.errno = (short) 1064;
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
        return (byte) Fields.FIELD_TYPE_VAR_STRING;
    }

    public void close() {

    }
}