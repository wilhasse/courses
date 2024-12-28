import com.alibaba.polardbx.net.FrontendConnection;
import com.alibaba.polardbx.net.factory.FrontendConnectionFactory;

import java.nio.channels.SocketChannel;

public class SimpleSplitServer extends SimpleServer {

    private static final SimpleSplitServer INSTANCE_SPLIT = new SimpleSplitServer();

    public static SimpleServer getInstance() {
        return INSTANCE_SPLIT;
    }

    class DebugSplitConnectionFactory extends FrontendConnectionFactory {
        @Override
        protected FrontendConnection getConnection(SocketChannel channel) {
            System.out.println("Creating new connection for channel: " + channel);
            DebugConnection c = new DebugConnection(channel);
            c.setQueryHandler(new SimpleSplitQueryHandler(c));
            return c;
        }
    }

    @Override
    protected FrontendConnectionFactory createConnectionFactory() {
        return new DebugSplitConnectionFactory();
    }

    public static void main(String[] args) {
        try {
            getInstance().startup();

            System.out.println("Split Server started successfully, press Ctrl+C to stop");
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

