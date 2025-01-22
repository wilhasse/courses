import com.alibaba.polardbx.rpc.pool.XConnection;
import com.alibaba.polardbx.rpc.pool.XConnectionManager;
import com.alibaba.polardbx.rpc.result.XResult;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * A round-robin connection pool to multiple backend XConnection data sources.
 */
public class MultiServerConnectionPool {

    private static final int POOL_SIZE_PER_SERVER = 1;

    private final XConnectionManager manager = XConnectionManager.getInstance();
    private final Map<ServerInfo, List<XConnection>> serverConnections = new ConcurrentHashMap<>();
    private final List<ServerInfo> serverList = Collections.synchronizedList(new ArrayList<>());
    private final AtomicInteger nextServerIndex = new AtomicInteger(0);
    private final AtomicInteger nextConnectionIndex = new AtomicInteger(0);
    private final Object lock = new Object();

    public void addServer(ServerInfo info) {
        synchronized (lock) {
            if (!serverConnections.containsKey(info)) {
                try {
                    manager.initializeDataSource(
                        info.host,
                        info.port,
                        info.username,
                        info.password,
                        "test-instance"
                    );
                    List<XConnection> list = new ArrayList<>(POOL_SIZE_PER_SERVER);
                    for (int i = 0; i < POOL_SIZE_PER_SERVER; i++) {
                        list.add(createConnection(info));
                    }
                    serverConnections.put(info, list);
                    serverList.add(info);
                    System.out.println("Added backend server " + info.host + ":" + info.port);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }

    private XConnection createConnection(ServerInfo info) throws Exception {
        XConnection conn = manager.getConnection(
            info.host, info.port,
            info.username, info.password,
            info.defaultDB,
            30000L * 1000000L
        );
        conn.setStreamMode(true);

        // do "USE <db>"
        XResult r = conn.execQuery("USE " + info.defaultDB);
        while (r.next() != null) {
            // consume
        }
        return conn;
    }

    /**
     * Round-robin across servers, then round-robin connections on that server.
     */
    public XConnection getNextConnection() {
        synchronized (lock) {
            if (serverList.isEmpty()) {
                throw new RuntimeException("No servers in pool");
            }
            int sIndex = nextServerIndex.getAndIncrement() % serverList.size();
            ServerInfo info = serverList.get(sIndex);

            List<XConnection> conns = serverConnections.get(info);
            if (conns == null || conns.isEmpty()) {
                throw new RuntimeException("No connections for server " + info.host);
            }

            int cIndex = nextConnectionIndex.getAndIncrement() % conns.size();
            XConnection conn = conns.get(cIndex);

            // Validate or replace
            try {
                if (!isConnectionValid(conn)) {
                    conn = createConnection(info);
                    conns.set(cIndex, conn);
                }
            } catch (Exception e) {
                // Attempt to replace
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

    private boolean isConnectionValid(XConnection conn) {
        try {
            XResult r = conn.execQuery("SELECT 1");
            while (r.next() != null) { /* consume */ }
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    public void close() {
        synchronized (lock) {
            for (Map.Entry<ServerInfo, List<XConnection>> e : serverConnections.entrySet()) {
                ServerInfo si = e.getKey();
                for (XConnection xconn : e.getValue()) {
                    if (xconn != null) {
                        try {
                            xconn.close();
                        } catch (Exception ignored) {}
                    }
                }
                manager.deinitializeDataSource(si.host, si.port, si.username, si.password);
            }
        }
    }
}
