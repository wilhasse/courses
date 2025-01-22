// package com.myproject.handler; // (adjust your package as needed)

/**
 * Public class can be any one of these. 
 * Example: Make ServerInfo the public class, so it's accessible.
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
}

// The following classes have default (package-private) visibility 
// because we did not mark them public. They can be used by other classes 
// in the same package (like MyQueryHandler).
class ChunkResult {
    public final com.alibaba.polardbx.rpc.result.XResult result;
    public final java.util.List<java.util.List<Object>> rows;

    public ChunkResult(com.alibaba.polardbx.rpc.result.XResult result,
                       java.util.List<java.util.List<Object>> rows) {
        this.result = result;
        this.rows = rows;
    }
}

class ChunkIterator {
    private final java.util.List<java.util.List<Object>> rows;
    private int index = 0;

    public ChunkIterator(java.util.List<java.util.List<Object>> rows) {
        this.rows = rows;
    }

    public java.util.List<Object> current() {
        return rows.get(index);
    }

    public boolean hasNext() {
        return index < rows.size() - 1;
    }

    public void next() {
        index++;
    }
}
