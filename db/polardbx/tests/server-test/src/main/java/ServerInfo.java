import com.alibaba.polardbx.rpc.result.XResult;
import java.util.List;

/**
 * Simple POJO describing connection information for a backend server.
 */
public class ServerInfo {
    public final String host;
    public final int port;
    public final String username;
    public final String password;
    public final String defaultDB;

    public ServerInfo(String h, int p, String u, String pass, String db) {
        this.host = h;
        this.port = p;
        this.username = u;
        this.password = pass;
        this.defaultDB = db;
    }

    // Optional: override equals and hashCode if needed for the map keys
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof ServerInfo)) return false;
        ServerInfo that = (ServerInfo) o;
        return port == that.port
            && host.equals(that.host)
            && username.equals(that.username)
            && password.equals(that.password)
            && defaultDB.equals(that.defaultDB);
    }

    @Override
    public int hashCode() {
        int result = host.hashCode();
        result = 31 * result + port;
        result = 31 * result + username.hashCode();
        result = 31 * result + password.hashCode();
        result = 31 * result + defaultDB.hashCode();
        return result;
    }
}

/**
 * Holds the result of a chunk execution: the XResult for metadata,
 * plus a list of fully-read rows.
 */
class ChunkResult {
    public final XResult result;
    public final List<List<Object>> rows;

    public ChunkResult(XResult result, List<List<Object>> rows) {
        this.result = result;
        this.rows = rows;
    }
}

/**
 * A helper class that iterates over a list of rows from a single chunk.
 * Used by the PriorityQueue for merging.
 */
class ChunkIterator {
    private final List<List<Object>> rows;
    private int index = 0;

    /**
     * @param rows All rows for one chunk
     */
    public ChunkIterator(List<List<Object>> rows) {
        this.rows = rows;
    }

    /**
     * @return The current row (List of column values)
     */
    public List<Object> current() {
        return rows.get(index);
    }

    /**
     * @return true if there's another row after the current one
     */
    public boolean hasNext() {
        return index < rows.size() - 1;
    }

    /**
     * Move to the next row.
     */
    public void next() {
        index++;
    }
}
