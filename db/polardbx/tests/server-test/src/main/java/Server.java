import com.alibaba.polardbx.net.NIOAcceptor;
import com.alibaba.polardbx.net.NIOProcessor;
import com.alibaba.polardbx.net.factory.FrontendConnectionFactory;
import com.alibaba.polardbx.net.util.TimeUtil;
import com.alibaba.polardbx.common.utils.thread.ServerThreadPool;
import com.alibaba.polardbx.common.utils.thread.ThreadCpuStatUtil;

import java.io.IOException;
import java.nio.channels.SocketChannel;
import java.util.Timer;
import java.util.TimerTask;

/**
 * Standalone server for handling MySQL-protocol connections.
 * Creates multiple processors, sets up the NIOAcceptor on a given port,
 * and uses a custom FrontendConnectionFactory that produces SplitConnections.
 */
public class Server {

    private static final int SERVER_PORT = 8507;
    private NIOProcessor[] processors;
    private NIOAcceptor acceptor;

    /**
     * A factory that creates a SplitConnection and sets its handler to SplitQueryHandler.
     */
    class SplitConnectionFactory extends FrontendConnectionFactory {
        @Override
        protected com.alibaba.polardbx.net.FrontendConnection getConnection(SocketChannel channel) {
            System.out.println("Accepted a new channel: " + channel);
            ServerConnection conn = new ServerConnection(channel);
            // Our single query handler that does parallel-splitting
            conn.setQueryHandler(new MyQueryHandler(conn));
            return conn;
        }
    }

    /**
     * Starts the NIO-based server, listening on SERVER_PORT.
     */
    public void startup() throws IOException {
        System.out.println("Starting Server on port " + SERVER_PORT);

        // Periodically update server time
        Timer timer = new Timer("ServerTimeUpdater", true);
        timer.schedule(new TimerTask() {
            @Override
            public void run() {
                TimeUtil.update();
            }
        }, 0L, 100L);

        // Create NIOProcessors (1 per CPU core, or at least 1)
        int cores = Math.max(1, ThreadCpuStatUtil.NUM_CORES);
        this.processors = new NIOProcessor[cores];
        for (int i = 0; i < cores; i++) {
            ServerThreadPool handlerPool = new ServerThreadPool(
                "ProcessorHandler-" + i,
                4,         // pool size
                5000,      // deadLockCheckPeriod ms
                1          // bucketSize
            );
            processors[i] = new NIOProcessor(i, "Processor" + i, handlerPool);
            processors[i].startup();
            System.out.println("Processor " + i + " started.");
        }

        // Create the acceptor
        FrontendConnectionFactory factory = new SplitConnectionFactory();
        acceptor = new NIOAcceptor("SplitAcceptor", SERVER_PORT, factory, true);
        acceptor.setProcessors(processors);
        acceptor.start();

        System.out.println("Server fully started on port " + SERVER_PORT);
    }

    /**
     * Main entry point. Launches the server, then loops.
     */
    public static void main(String[] args) {
        try {
            Server server = new Server();
            server.startup();

            System.out.println("Server started successfully. Press Ctrl+C to stop.");
            for (;;) {
                Thread.sleep(1000);
            }
        } catch (Exception e) {
            System.err.println("Failed to start server: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}
