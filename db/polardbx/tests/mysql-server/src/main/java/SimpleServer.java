import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.SelectionKey;
import java.nio.channels.ServerSocketChannel;
import java.nio.channels.Selector;
import java.nio.channels.SocketChannel;
import java.nio.charset.StandardCharsets;
import java.util.Iterator;
import java.util.Properties;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class SimpleServer {
    private final int port;
    private final Properties systemConfig;
    private ServerSocketChannel serverChannel;
    private Selector selector;
    private ExecutorService serverExecutor;
    private boolean running;

    // MySQL protocol constants
    private static final int PROTOCOL_VERSION = 10;
    private static final String SERVER_VERSION = "5.7.0-SimpleServer";
    private static final int CLIENT_PROTOCOL_41 = 0x00000512;
    private static final int CLIENT_CONNECT_WITH_DB = 0x00000008;
    private static final int CLIENT_PLUGIN_AUTH = 0x00080000;
    private static final byte OK_PACKET = 0x00;
    private static final byte ERROR_PACKET = (byte) 0xff;
  
    // Client capabilities
    private static final int CLIENT_LONG_PASSWORD = 1;
    private static final int CLIENT_FOUND_ROWS = 2;
    private static final int CLIENT_LONG_FLAG = 4;
    private static final int CLIENT_NO_SCHEMA = 16;
    private static final int CLIENT_INTERACTIVE = 1024;
    private static final int CLIENT_IGNORE_SPACE = 256;
    private static final int CLIENT_LOCAL_FILES = 128;
    private static final int CLIENT_IGNORE_SIGPIPE = 4096;
    private static final int CLIENT_TRANSACTIONS = 8192;
    
    private static final int SERVER_STATUS_AUTOCOMMIT = 0x0002;

    // Combined server capabilities (explicitly not including SSL)
    private static final int SERVER_CAPABILITIES = 
        CLIENT_LONG_PASSWORD |
        CLIENT_FOUND_ROWS |
        CLIENT_LONG_FLAG |
        CLIENT_CONNECT_WITH_DB |
        CLIENT_NO_SCHEMA |
        CLIENT_PROTOCOL_41 |
        CLIENT_INTERACTIVE |
        CLIENT_IGNORE_SPACE |
        CLIENT_LOCAL_FILES |
        CLIENT_IGNORE_SIGPIPE |
        CLIENT_TRANSACTIONS;
	
    public SimpleServer(int port) {
        this.port = port;
        this.systemConfig = new Properties();
        initializeDefaultConfig();
    }

    private void initializeDefaultConfig() {
        systemConfig.setProperty("serverPort", String.valueOf(port));
        systemConfig.setProperty("processors", String.valueOf(Runtime.getRuntime().availableProcessors()));
        systemConfig.setProperty("processorHandler", "4");
        systemConfig.setProperty("processorExecutor", "4");
    }

    public void start() throws Exception {
        initializeSystemComponents();
        createThreadPools();
        initializeNetworkLayer();
        startServer();
        System.out.println("SimpleServer started on port " + port);
    }

    private void initializeSystemComponents() {
        System.out.println("Initializing system components...");
    }

    private void createThreadPools() {
        int processors = Integer.parseInt(systemConfig.getProperty("processors"));
        serverExecutor = Executors.newFixedThreadPool(processors);
        System.out.println("Created server executor with " + processors + " threads");
    }

    private void initializeNetworkLayer() throws IOException {
        selector = Selector.open();
        serverChannel = ServerSocketChannel.open();
        serverChannel.configureBlocking(false);
        serverChannel.socket().bind(new InetSocketAddress(port));
        serverChannel.register(selector, SelectionKey.OP_ACCEPT);
        System.out.println("Network layer initialized");
    }

    private void startServer() {
        running = true;
        new Thread(this::acceptLoop, "MySQL-Acceptor").start();
    }

    private void acceptLoop() {
        while (running) {
            try {
                if (selector.select() > 0) {
                    Iterator<SelectionKey> keys = selector.selectedKeys().iterator();
                    while (keys.hasNext()) {
                        SelectionKey key = keys.next();
                        keys.remove();

                        if (!key.isValid()) {
                            continue;
                        }

                        if (key.isAcceptable()) {
                            accept(key);
                        } else if (key.isReadable()) {
                            read(key);
                        }
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    private void accept(SelectionKey key) throws IOException {
        ServerSocketChannel serverChannel = (ServerSocketChannel) key.channel();
        SocketChannel clientChannel = serverChannel.accept();
        clientChannel.configureBlocking(false);
        clientChannel.register(selector, SelectionKey.OP_READ);
        
        sendHandshakePacket(clientChannel);
        System.out.println("New connection accepted");
    }

    private void read(SelectionKey key) {
        SocketChannel channel = (SocketChannel) key.channel();
        ByteBuffer buffer = ByteBuffer.allocate(1024);
        
        try {
            int bytesRead = channel.read(buffer);
            if (bytesRead == -1) {
                closeConnection(key);
                return;
            }
            
            buffer.flip();
            
            // Print received packet for debugging
            System.out.println("Received packet of " + bytesRead + " bytes");
            printPacketHex(buffer);
            
            byte commandType = buffer.get(4);
            System.out.println("Command type: 0x" + String.format("%02x", commandType));
            
            if (commandType == 0x1) { // Login packet
                handleLoginPacket(channel, buffer);
            } else if (commandType == 0x3) { // Query packet
                handleQueryPacket(channel, buffer);
            } else {
                System.out.println("Unknown command type: 0x" + String.format("%02x", commandType));
                sendOkPacket(channel);
            }
            
        } catch (IOException e) {
            System.out.println("Error reading from channel: " + e.getMessage());
            closeConnection(key);
        }
    }

    private void handleLoginPacket(SocketChannel channel, ByteBuffer buffer) throws IOException {
        System.out.println("Processing login packet");
        sendOkPacket(channel);
        System.out.println("Sent OK packet for login");
    }

    private String extractQuery(ByteBuffer buffer) {
        byte[] queryBytes = new byte[buffer.remaining() - 5];
        buffer.position(5);
        buffer.get(queryBytes);
        return new String(queryBytes, StandardCharsets.UTF_8);
    }

// Then modify the sendHandshakePacket method to use these capabilities:
    private void sendHandshakePacket(SocketChannel channel) throws IOException {
        ByteBuffer buffer = ByteBuffer.allocate(128);
        
        // Packet header (will be filled in later)
        buffer.put((byte) 0);  // length LSB
        buffer.put((byte) 0);  // length MSB
        buffer.put((byte) 0);  // length top
        buffer.put((byte) 0);  // sequence number
        
        // Protocol version
        buffer.put((byte) PROTOCOL_VERSION);
        
        // Server version
        buffer.put(SERVER_VERSION.getBytes());
        buffer.put((byte) 0);
        
        // Connection id
        buffer.putInt(1234);
        
        // Auth plugin data part 1
        byte[] authPluginData = new byte[20];
        for (int i = 0; i < 8; i++) {
            authPluginData[i] = (byte) (Math.random() * 255);
            buffer.put(authPluginData[i]);
        }
        
        buffer.put((byte) 0);  // filler
        
        // Capability flags (lower 2 bytes)
        buffer.putShort((short) (SERVER_CAPABILITIES & 0xffff));
        
        // Character set (utf8_general_ci)
        buffer.put((byte) 33);
        
        // Status flags
        buffer.putShort((short) SERVER_STATUS_AUTOCOMMIT);
        
        // Capability flags (upper 2 bytes)
        buffer.putShort((short) (SERVER_CAPABILITIES >> 16));
        
        // Auth plugin data length
        buffer.put((byte) 21);
        
        // Reserved (10 bytes)
        for (int i = 0; i < 10; i++) {
            buffer.put((byte) 0);
        }
        
        // Auth plugin data part 2
        for (int i = 8; i < 20; i++) {
            authPluginData[i] = (byte) (Math.random() * 255);
            buffer.put(authPluginData[i]);
        }
        buffer.put((byte) 0);
        
        // Auth plugin name
        buffer.put("mysql_native_password".getBytes());
        buffer.put((byte) 0);
        
        // Write packet length
        int packetLength = buffer.position() - 4;
        buffer.put(0, (byte) (packetLength & 0xff));
        buffer.put(1, (byte) (packetLength >> 8));
        buffer.put(2, (byte) (packetLength >> 16));
        
        buffer.flip();
        channel.write(buffer);
        System.out.println("Sent handshake packet without SSL capability");
    }

// Update the handleSelect1Query method in SimpleServer.java
    private void handleSelect1Query(SocketChannel channel) throws IOException {
        // 1. Send column count (1 column)
        ByteBuffer buffer = ByteBuffer.allocate(5);
        buffer.put((byte) 1);  // length
        buffer.put((byte) 0);
        buffer.put((byte) 0);
        buffer.put((byte) 1);  // sequence number
        buffer.put((byte) 1);  // column count = 1
        buffer.flip();
        channel.write(buffer);

        // 2. Send column definition
        buffer = ByteBuffer.allocate(128);
        buffer.put((byte) 0);  // length placeholder
        buffer.put((byte) 0);
        buffer.put((byte) 0);
        buffer.put((byte) 2);  // sequence number

        // Column Definition
        writeString(buffer, "def");           // catalog
        writeString(buffer, "");              // schema
        writeString(buffer, "");              // table
        writeString(buffer, "");              // org_table
        writeString(buffer, "1");             // name
        writeString(buffer, "");              // org_name
        buffer.put((byte) 0x0c);             // length of fixed fields
        buffer.putShort((short) 63);         // character set (utf8_general_ci)
        buffer.putInt(1);                    // column length
        buffer.put((byte) 0x03);             // column type (MYSQL_TYPE_LONG)
        buffer.putShort((short) 0);          // flags
        buffer.put((byte) 0);                // decimals
        buffer.putShort((short) 0);          // filler

        // Set packet length
        int length = buffer.position() - 4;
        buffer.put(0, (byte) (length & 0xff));
        buffer.put(1, (byte) (length >> 8));
        buffer.put(2, (byte) (length >> 16));
        buffer.flip();
        channel.write(buffer);

        // 3. Send EOF packet
        buffer = ByteBuffer.allocate(9);
        buffer.put((byte) 5);    // length
        buffer.put((byte) 0);
        buffer.put((byte) 0);
        buffer.put((byte) 3);    // sequence number
        buffer.put((byte) 0xfe); // EOF marker
        buffer.putShort((short) 0);  // warnings
        buffer.putShort((short) 2);  // status flags
        buffer.flip();
        channel.write(buffer);

        // 4. Send row data
        buffer = ByteBuffer.allocate(64);
        buffer.put((byte) 0);    // length placeholder
        buffer.put((byte) 0);
        buffer.put((byte) 0);
        buffer.put((byte) 4);    // sequence number
        writeString(buffer, "1"); // row data
        length = buffer.position() - 4;
        buffer.put(0, (byte) (length & 0xff));
        buffer.put(1, (byte) (length >> 8));
        buffer.put(2, (byte) (length >> 16));
        buffer.flip();
        channel.write(buffer);

        // 5. Send EOF packet
        buffer = ByteBuffer.allocate(9);
        buffer.put((byte) 5);    // length
        buffer.put((byte) 0);
        buffer.put((byte) 0);
        buffer.put((byte) 5);    // sequence number
        buffer.put((byte) 0xfe); // EOF marker
        buffer.putShort((short) 0);  // warnings
        buffer.putShort((short) 2);  // status flags
        buffer.flip();
        channel.write(buffer);
    }

    // Update the handleQueryPacket method to handle version query
    private void handleQueryPacket(SocketChannel channel, ByteBuffer buffer) throws IOException {
        String query = extractQuery(buffer);
        System.out.println("Received query: " + query);
        
        if (query.trim().equalsIgnoreCase("select 1")) {
            handleSelect1Query(channel);
        } else if (query.trim().equalsIgnoreCase("select @@version_comment limit 1")) {
            handleVersionQuery(channel);
        } else {
            sendOkPacket(channel);
        }
    }

    // Add method to handle version query
    private void handleVersionQuery(SocketChannel channel) throws IOException {
        // Similar structure to handleSelect1Query but returning version info
        // 1. Column count
        ByteBuffer buffer = ByteBuffer.allocate(5);
        buffer.put((byte) 1);
        buffer.put((byte) 0);
        buffer.put((byte) 0);
        buffer.put((byte) 1);
        buffer.put((byte) 1);
        buffer.flip();
        channel.write(buffer);

        // 2. Column definition
        buffer = ByteBuffer.allocate(128);
        buffer.put((byte) 0);
        buffer.put((byte) 0);
        buffer.put((byte) 0);
        buffer.put((byte) 2);

        writeString(buffer, "def");
        writeString(buffer, "");
        writeString(buffer, "");
        writeString(buffer, "");
        writeString(buffer, "@@version_comment");
        writeString(buffer, "");
        buffer.put((byte) 0x0c);
        buffer.putShort((short) 63);
        buffer.putInt(80);
        buffer.put((byte) 0xfd);  // MYSQL_TYPE_VAR_STRING
        buffer.putShort((short) 0);
        buffer.put((byte) 0);
        buffer.putShort((short) 0);

        int length = buffer.position() - 4;
        buffer.put(0, (byte) (length & 0xff));
        buffer.put(1, (byte) (length >> 8));
        buffer.put(2, (byte) (length >> 16));
        buffer.flip();
        channel.write(buffer);

        // 3. EOF
        sendEofPacket(channel, (byte) 3);

        // 4. Row data
        buffer = ByteBuffer.allocate(64);
        buffer.put((byte) 0);
        buffer.put((byte) 0);
        buffer.put((byte) 0);
        buffer.put((byte) 4);
        writeString(buffer, "Simple MySQL Protocol Implementation");

        length = buffer.position() - 4;
        buffer.put(0, (byte) (length & 0xff));
        buffer.put(1, (byte) (length >> 8));
        buffer.put(2, (byte) (length >> 16));
        buffer.flip();
        channel.write(buffer);

        // 5. EOF
        sendEofPacket(channel, (byte) 5);
    }
	
    private void sendColumnCount(SocketChannel channel, int count) throws IOException {
        ByteBuffer buffer = ByteBuffer.allocate(5);
        buffer.put((byte) 1);
        buffer.put((byte) 0);
        buffer.put((byte) 0);
        buffer.put((byte) 1);
        buffer.put((byte) count);
        buffer.flip();
        channel.write(buffer);
    }

    private void sendColumnDefinition(SocketChannel channel, String name, String type) throws IOException {
        ByteBuffer buffer = ByteBuffer.allocate(128);
        
        int startPos = buffer.position() + 4;
        
        writeString(buffer, "def");
        writeString(buffer, "");
        writeString(buffer, "");
        writeString(buffer, "");
        writeString(buffer, name);
        writeString(buffer, "");
        
        buffer.put((byte) 0x0c);
        buffer.putShort((short) 33);
        buffer.putInt(20);
        buffer.put((byte) 8);
        buffer.putShort((short) 0x8001);
        buffer.put((byte) 0);
        buffer.putShort((short) 0);
        
        int packetLength = buffer.position() - startPos;
        buffer.put(0, (byte) (packetLength & 0xff));
        buffer.put(1, (byte) (packetLength >> 8));
        buffer.put(2, (byte) (packetLength >> 16));
        buffer.put(3, (byte) 2);
        
        buffer.flip();
        channel.write(buffer);
    }

    private void sendEofPacket(SocketChannel channel, byte sequenceNumber) throws IOException {
        ByteBuffer buffer = ByteBuffer.allocate(9);
        
        buffer.put((byte) 5);
        buffer.put((byte) 0);
        buffer.put((byte) 0);
        buffer.put(sequenceNumber);
        buffer.put((byte) 0xfe);
        buffer.putShort((short) 0);
        buffer.putShort((short) 0x0002);
        
        buffer.flip();
        channel.write(buffer);
    }

    private void sendRowData(SocketChannel channel, String value) throws IOException {
        ByteBuffer buffer = ByteBuffer.allocate(128);
        
        int startPos = buffer.position() + 4;
        writeString(buffer, value);
        
        int packetLength = buffer.position() - startPos;
        buffer.put(0, (byte) (packetLength & 0xff));
        buffer.put(1, (byte) (packetLength >> 8));
        buffer.put(2, (byte) (packetLength >> 16));
        buffer.put(3, (byte) 3);
        
        buffer.flip();
        channel.write(buffer);
    }

    private void sendOkPacket(SocketChannel channel) throws IOException {
        ByteBuffer buffer = ByteBuffer.allocate(11);
        
        buffer.put((byte) 7);
        buffer.put((byte) 0);
        buffer.put((byte) 0);
        buffer.put((byte) 2);
        buffer.put(OK_PACKET);
        buffer.put((byte) 0);
        buffer.put((byte) 0);
        buffer.putShort((short) 0x0002);
        buffer.putShort((short) 0);
        
        buffer.flip();
        channel.write(buffer);
    }

    private void writeString(ByteBuffer buffer, String str) {
        byte[] bytes = str.getBytes(StandardCharsets.UTF_8);
        buffer.put((byte) bytes.length);
        buffer.put(bytes);
    }

    private void printPacketHex(ByteBuffer buffer) {
        byte[] data = new byte[buffer.remaining()];
        buffer.mark();
        buffer.get(data);
        buffer.reset();
        
        StringBuilder sb = new StringBuilder();
        for (byte b : data) {
            sb.append(String.format("%02x ", b));
        }
        System.out.println("Packet hex dump: " + sb.toString());
    }

    private void closeConnection(SelectionKey key) {
        try {
            key.channel().close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        key.cancel();
    }

    public void stop() {
        running = false;
        try {
            if (serverChannel != null) {
                serverChannel.close();
            }
            if (selector != null) {
                selector.close();
            }
            if (serverExecutor != null) {
                serverExecutor.shutdown();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Server stopped");
    }

    public static void main(String[] args) {
        try {
            SimpleServer server = new SimpleServer(3306);
            server.start();
            
            // Keep the server running
            Thread.currentThread().join();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}