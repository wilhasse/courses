import com.alibaba.polardbx.common.model.DbPriv;
import com.alibaba.polardbx.common.utils.logger.Logger;
import com.alibaba.polardbx.common.utils.logger.LoggerFactory;
import com.alibaba.polardbx.net.FrontendConnection;
import com.alibaba.polardbx.net.buffer.BufferPool;
import com.alibaba.polardbx.net.buffer.ByteBufferHolder;
import com.alibaba.polardbx.net.handler.Privileges;
import com.taobao.tddl.common.privilege.EncrptPassword;

import java.util.Map;
import java.util.HashMap;
import java.util.Set;
import java.nio.ByteBuffer;
import java.nio.channels.SocketChannel;
import java.util.concurrent.atomic.AtomicLong;

public class DebugConnection extends FrontendConnection {
    private final Logger logger = LoggerFactory.getLogger(DebugConnection.class);
    private final BufferPool bufferPool;
    private final AtomicLong CONNECTION_ID = new AtomicLong(1);
    private final long connectionId;
    private SimpleConfig config;

    public DebugConnection(SocketChannel channel) {
        super(channel);
        this.config = new SimpleConfig();
        this.bufferPool = new BufferPool(1024 * 1024 * 16, 4096);
        this.packetHeaderSize = 4;
        this.maxPacketSize = 16 * 1024 * 1024;
        this.readBuffer = allocate();
        this.connectionId = CONNECTION_ID.getAndIncrement();
        this.privileges = new SimplePrivileges();

        System.out.println("Created new connection " + connectionId + " with buffer pool");
        System.out.println("Created new connection with buffer pool");
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
        public Map<String, DbPriv> getSchemaPrivs(String user, String host) {
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

    @Override
    protected void cleanup() {
        try {
            if (queryHandler instanceof SimpleQueryHandler) {
                ((SimpleQueryHandler) queryHandler).close();
            }
        } catch (Exception e) {
            logger.error("Error closing query handler", e);
        }
        super.cleanup();
    }

    public SimpleConfig getConfig() {
        return config;
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


