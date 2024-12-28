import com.alibaba.polardbx.net.FrontendConnection;
import com.alibaba.polardbx.net.NIOAcceptor;
import com.alibaba.polardbx.net.NIOProcessor;
import com.alibaba.polardbx.net.factory.FrontendConnectionFactory;
import com.alibaba.polardbx.net.util.TimeUtil;
import com.alibaba.polardbx.common.utils.thread.ThreadCpuStatUtil;
import com.alibaba.polardbx.common.utils.thread.ServerThreadPool;
import java.io.IOException;
import java.nio.channels.SocketChannel;
import java.util.Timer;
import java.util.TimerTask;

public class SimpleServer {
    private static final SimpleServer INSTANCE = new SimpleServer();
    protected static final int SERVER_PORT = 8507;
    protected NIOProcessor[] processors;
    protected NIOAcceptor server;

    public static SimpleServer getInstance() {
        return INSTANCE;
    }

    class DebugConnectionFactory extends FrontendConnectionFactory {
        @Override
        protected FrontendConnection getConnection(SocketChannel channel) {
            System.out.println("Creating new connection for channel: " + channel);
            DebugConnection c = new DebugConnection(channel);
            c.setQueryHandler(new SimpleQueryHandler(c));
            return c;
        }
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
        FrontendConnectionFactory factory = createConnectionFactory();
        server = new NIOAcceptor("MySQLServer", SERVER_PORT, factory, true);
        server.setProcessors(processors);
        server.start();

        System.out.println("Server started on port " + SERVER_PORT);
    }

    protected FrontendConnectionFactory createConnectionFactory() {
        return new DebugConnectionFactory();
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
