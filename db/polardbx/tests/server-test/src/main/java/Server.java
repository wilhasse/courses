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
 * Standalone server for handling MySQL-protocol connections via the polardbx-net framework.
 * <p>
 * This class bootstraps NIOProcessors and a single NIOAcceptor to listen on a specific port.
 * A custom FrontendConnectionFactory (SplitConnectionFactory) is provided, which creates our
 * ServerConnection instances using the MyQueryHandler for parallel-splitting logic.
 */
public class Server {

    private static final int SERVER_PORT = 8507;
    private NIOProcessor[] processors;
    private NIOAcceptor acceptor;

    /**
     * A factory that creates a ServerConnection and sets its handler to MyQueryHandler.
     */
    class SplitConnectionFactory extends FrontendConnectionFactory {
        @Override
        protected com.alibaba.polardbx.net.FrontendConnection getConnection(SocketChannel channel) {
            System.out.println("Accepted a new channel: " + channel);

            // Build our custom ServerConnection
            ServerConnection conn = new ServerConnection(channel);

            // Assign the parallel-splitting query handler to it
            conn.setQueryHandler(new MyQueryHandler(conn));

            return conn;
        }
    }

    /**
     * Starts the NIO-based server, listening on SERVER_PORT.
     *
     * @throws IOException if server fails to bind or start
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
                4,         // example thread pool size
                5000,      // deadLockCheckPeriod ms
                1          // bucketSize
            );
            processors[i] = new NIOProcessor(i, "Processor" + i, handlerPool);
            processors[i].startup();
            System.out.println("Processor " + i + " started.");
        }

        // Create the acceptor, specifying our custom connection factory
        FrontendConnectionFactory factory = new SplitConnectionFactory();
        acceptor = new NIOAcceptor("SplitAcceptor", SERVER_PORT, factory, true);
        acceptor.setProcessors(processors);
        acceptor.start();

        System.out.println("Server fully started on port " + SERVER_PORT);
    }

    /**
     * Main entry point. Launches the server and loops forever.
     */
    public static void main(String[] args) {
        try {
            Server server = new Server();
            server.startup();

            System.out.println("Server started successfully. Press Ctrl+C to stop.");
            // Prevent exit
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
