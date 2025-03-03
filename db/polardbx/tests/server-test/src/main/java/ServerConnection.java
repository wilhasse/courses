import com.alibaba.polardbx.common.model.DbPriv;
import com.alibaba.polardbx.common.utils.logger.Logger;
import com.alibaba.polardbx.common.utils.logger.LoggerFactory;
import com.alibaba.polardbx.net.FrontendConnection;
import com.alibaba.polardbx.net.buffer.BufferPool;
import com.alibaba.polardbx.net.buffer.ByteBufferHolder;
import com.alibaba.polardbx.net.handler.Privileges;
import com.taobao.tddl.common.privilege.EncrptPassword;

import java.nio.ByteBuffer;
import java.nio.channels.SocketChannel;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Custom connection class that extends FrontendConnection (required by the polardbx-net framework).
 * <p>
 * Manages user credentials, buffer allocation, and delegates queries to MyQueryHandler.
 */
public class ServerConnection extends FrontendConnection {
    private static final Logger logger = LoggerFactory.getLogger(ServerConnection.class);

    private static final AtomicLong CONN_ID_GENERATOR = new AtomicLong(1);

    private final BufferPool bufferPool;
    private final long connectionId;
    private final Config config;

    /**
     * Constructs a new ServerConnection with a dedicated buffer pool and a unique ID.
     *
     * @param channel The SocketChannel for client communication
     */
    public ServerConnection(SocketChannel channel) {
        super(channel);

        // A simple buffer pool. Adjust size as appropriate for your environment.
        this.bufferPool = new BufferPool(16 * 1024 * 1024, 4096);
        this.packetHeaderSize = 4;
        this.maxPacketSize = 16 * 1024 * 1024;
        this.readBuffer = allocate();

        // Generate a unique connection ID
        this.connectionId = CONN_ID_GENERATOR.getAndIncrement();

        // Minimal privileges object plus user config
        this.privileges = new SimplePrivileges();
        this.config = new Config();

        System.out.println("Created new ServerConnection #" + connectionId);
    }

    /**
     * Basic config holding user/password info for authentication.
     */
    static class Config {
        private final Map<String, String> users = new HashMap<>();
        public Config() {
            // Hardcoded user/password for demonstration
            users.put("root", "12345");
        }
        public Map<String, String> getUsers() {
            return users;
        }
    }

    /**
     * Minimal privileges implementation for demonstration only.
     */
    class SimplePrivileges implements Privileges {
        @Override public boolean schemaExists(String schema) { return true; }
        @Override public boolean userExists(String user) {
            return config.getUsers().containsKey(user);
        }
        @Override public boolean userExists(String user, String host) { return userExists(user); }
        @Override public boolean userMatches(String user, String host) { return true; }
        @Override public EncrptPassword getPassword(String user) {
            String pass = config.getUsers().get(user);
            return new EncrptPassword(pass, false);
        }
        @Override public EncrptPassword getPassword(String user, String host) {
            return getPassword(user);
        }
        @Override public Set<String> getUserSchemas(String user) { return null; }
        @Override public Set<String> getUserSchemas(String user, String host) { return null; }
        @Override public boolean isTrustedIp(String host, String user) { return true; }
        @Override public Map<String, DbPriv> getSchemaPrivs(String user, String host) { return null; }
        @Override
        public Map<String, com.alibaba.polardbx.common.model.TbPriv> getTablePrivs(String user, String host, String db) {
            return null;
        }
        @Override public boolean checkQuarantine(String user, String host) { return true; }
    }

    public Config getConfig() {
        return config;
    }

    /**
     * Allocates a new ByteBufferHolder from our buffer pool.
     *
     * @return The allocated holder, or ByteBufferHolder.EMPTY if allocation fails
     */
    @Override
    public ByteBufferHolder allocate() {
        try {
            ByteBufferHolder holder = bufferPool.allocate();
            if (holder != null) {
                ByteBuffer buf = holder.getBuffer();
                buf.clear();
                buf.position(0);
                buf.limit(buf.capacity());
                return holder;
            }
        } catch (Exception e) {
            logger.error("Error allocating buffer", e);
        }
        return ByteBufferHolder.EMPTY;
    }

    /**
     * Recycles a buffer back into the pool.
     *
     * @param buffer The ByteBufferHolder to recycle
     */
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
    protected long genConnId() {
        return connectionId;
    }

    @Override
    public long getId() {
        return connectionId;
    }

    /**
     * Cleans up resources when connection is closed, including the query handler if it is MyQueryHandler.
     */
    @Override
    protected void cleanup() {
        try {
            if (queryHandler instanceof MyQueryHandler) {
                ((MyQueryHandler) queryHandler).close();
            }
        } catch (Exception e) {
            logger.error("Error closing query handler", e);
        }
        super.cleanup();
    }

    /**
     * Called when there's a connection-related error.
     */
    @Override
    public void handleError(com.alibaba.polardbx.common.exception.code.ErrorCode errorCode, Throwable t) {
        logger.error("Connection error. Code=" + errorCode, t);
        close();
    }

    // Stub overrides to satisfy the abstract methods from FrontendConnection:
    @Override public boolean checkConnectionCount() { return true; }
    @Override public void addConnectionCount() {}
    @Override public boolean isPrivilegeMode() { return true; }
    @Override public boolean prepareLoadInfile(String sql) { return false; }
    @Override public void binlogDump(byte[] data) {}
    @Override public void fieldList(byte[] data) {}
}
