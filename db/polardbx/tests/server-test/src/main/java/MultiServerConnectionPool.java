import com.alibaba.polardbx.rpc.pool.XConnection;
import com.alibaba.polardbx.rpc.pool.XConnectionManager;
import com.alibaba.polardbx.rpc.result.XResult;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * A round-robin connection pool to multiple backend XConnection data sources.
 * <p>
 * This class manages multiple backend servers. Each server can have a small pool of
 * XConnection objects. Connections are retrieved in a round-robin fashion among servers,
 * and also within a server's connection list. If a connection is invalid, it is replaced.
 */
public class MultiServerConnectionPool {

    // Number of XConnections to maintain per server.
    private static final int POOL_SIZE_PER_SERVER = 1;

    // Singleton manager responsible for creating and managing XConnection objects.
    private final XConnectionManager manager = XConnectionManager.getInstance();

    /**
     * Maps a ServerInfo to a list of XConnection objects for that server.
     * Using ConcurrentHashMap for thread-safe read/write.
     */
    private final Map<ServerInfo, List<XConnection>> serverConnections = new ConcurrentHashMap<>();

    /**
     * Keeps a synchronized list of all servers currently in the pool.
     * We use a synchronizedList so that multiple threads can see consistent additions/removals.
     */
    private final List<ServerInfo> serverList = Collections.synchronizedList(new ArrayList<>());

    // Atomic counters for round-robin selection.
    private final AtomicInteger nextServerIndex = new AtomicInteger(0);
    private final AtomicInteger nextConnectionIndex = new AtomicInteger(0);

    // A lock object for synchronized code blocks.
    private final Object lock = new Object();

    /**
     * Add a new server to the connection pool. If the server is already present, do nothing.
     *
     * @param info The ServerInfo describing host, port, credentials, etc.
     */
    public void addServer(ServerInfo info) {
        synchronized (lock) {
            if (!serverConnections.containsKey(info)) {
                try {
                    // Initialize a data source in XConnectionManager for the new server
                    manager.initializeDataSource(
                        info.host,
                        info.port,
                        info.username,
                        info.password,
                        "test-instance"
                    );

                    // Create a list of connections for this server
                    List<XConnection> list = new ArrayList<>(POOL_SIZE_PER_SERVER);
                    for (int i = 0; i < POOL_SIZE_PER_SERVER; i++) {
                        list.add(createConnection(info));
                    }

                    // Store the connection list and add server to the round-robin list
                    serverConnections.put(info, list);
                    serverList.add(info);

                    System.out.println("Added backend server " + info.host + ":" + info.port);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }

    /**
     * Creates a new physical connection to the server indicated by info.
     *
     * @param info Server information such as host, port, etc.
     * @return A newly created XConnection object
     * @throws Exception If connection fails
     */
    private XConnection createConnection(ServerInfo info) throws Exception {
        // Acquire a connection from the XConnectionManager
        XConnection conn = manager.getConnection(
            info.host, info.port,
            info.username, info.password,
            info.defaultDB,
            30000L * 1000000L // A sample long timeout value (converted from ms to ns if needed)
        );
        // Enable streaming mode for large result sets
        conn.setStreamMode(true);

        // Example: send a "USE <db>" command to select the default database
        XResult r = conn.execQuery("USE " + info.defaultDB);
        while (r.next() != null) {
            // consume result (do nothing here)
        }
        return conn;
    }

    /**
     * Retrieves the next connection from the pool in a round-robin fashion.
     * 1. Round-robin among serverList.
     * 2. Round-robin among that server's connection list.
     * 3. Validate the connection; if it's invalid, replace it.
     *
     * @return A valid XConnection
     */
    public XConnection getNextConnection() {
        synchronized (lock) {
            if (serverList.isEmpty()) {
                throw new RuntimeException("No servers in pool");
            }
            // Pick the next server index (round-robin).
            int sIndex = nextServerIndex.getAndIncrement() % serverList.size();
            ServerInfo info = serverList.get(sIndex);

            // Retrieve the connection list for that server
            List<XConnection> conns = serverConnections.get(info);
            if (conns == null || conns.isEmpty()) {
                throw new RuntimeException("No connections for server " + info.host);
            }

            // Pick the next connection index (round-robin).
            int cIndex = nextConnectionIndex.getAndIncrement() % conns.size();
            XConnection conn = conns.get(cIndex);

            // Validate or replace
            try {
                if (!isConnectionValid(conn)) {
                    // If not valid, recreate it
                    conn = createConnection(info);
                    conns.set(cIndex, conn);
                }
            } catch (Exception e) {
                // Attempt to replace if validation check or creation failed
                try {
                    conn = createConnection(info);
                    conns.set(cIndex, conn);
                } catch (Exception e2) {
                    throw new RuntimeException("Failed to create replacement conn: " + e2.getMessage());
                }
            }
            return conn;
        }
    }

    /**
     * Checks a connection's validity by running a simple query.
     *
     * @param conn The XConnection to test
     * @return true if valid, false otherwise
     */
    private boolean isConnectionValid(XConnection conn) {
        try {
            XResult r = conn.execQuery("SELECT 1");
            while (r.next() != null) {
                // consume
            }
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Closes all connections in the pool and deinitializes the data sources.
     */
    public void close() {
        synchronized (lock) {
            for (Map.Entry<ServerInfo, List<XConnection>> e : serverConnections.entrySet()) {
                ServerInfo si = e.getKey();
                for (XConnection xconn : e.getValue()) {
                    if (xconn != null) {
                        try {
                            xconn.close();
                        } catch (Exception ignored) {
                        }
                    }
                }
                manager.deinitializeDataSource(si.host, si.port, si.username, si.password);
            }
        }
    }
}
