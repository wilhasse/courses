import com.alibaba.polardbx.rpc.result.XResult;
import com.alibaba.polardbx.rpc.compatible.XDataSource;
import com.alibaba.polardbx.rpc.result.XResultUtil;
import com.alibaba.polardbx.rpc.pool.XConnection;
import com.mysql.cj.polarx.protobuf.PolarxResultset;
import com.alibaba.polardbx.common.utils.Pair;
import com.google.protobuf.ByteString;

import java.util.ArrayList;
import java.util.List;

public class GalaxyTest {
    //public final static String SERVER_IP = "10.1.1.158";
    //public final static int SERVER_PORT = 32886;
    public final static String SERVER_IP = "10.1.1.148";
    public final static int SERVER_PORT = 33660;
    public final static String SERVER_USR = "teste";
    public final static String SERVER_PSW = "teste";
    private final static String DATABASE = "mysql";

    // Create a static dataSource with null properties string
    private static final XDataSource dataSource = 
        new XDataSource(SERVER_IP, SERVER_PORT, SERVER_USR, SERVER_PSW, DATABASE, "Test");

    public static void main(String[] args) throws Exception {
            GalaxyTest test = new GalaxyTest();
            test.playground();
    }

    public static XConnection getConn() throws Exception {
        return (XConnection) dataSource.getConnection();
    }

    public static List<List<Object>> getResult(XResult result) throws Exception {
        return getResult(result, false);
    }

    public static List<List<Object>> getResult(XResult result, boolean stringOrBytes) throws Exception {
        final List<PolarxResultset.ColumnMetaData> metaData = result.getMetaData();
        final List<List<Object>> ret = new ArrayList<>();
        while (result.next() != null) {
            final List<ByteString> data = result.current().getRow();
            assert metaData.size() == data.size();
            final List<Object> row = new ArrayList<>();
            for (int i = 0; i < metaData.size(); ++i) {
                final Pair<Object, byte[]> pair = XResultUtil
                    .resultToObject(metaData.get(i), data.get(i), true,
                        result.getSession().getDefaultTimezone());
                final Object obj =
                    stringOrBytes ? (pair.getKey() instanceof byte[] || null == pair.getValue() ? pair.getKey() :
                        new String(pair.getValue())) : pair.getKey();
                row.add(obj);
            }
            ret.add(row);
        }
        return ret;
    }

    private void show(XResult result) throws Exception {
        List<PolarxResultset.ColumnMetaData> metaData = result.getMetaData();
        for (PolarxResultset.ColumnMetaData meta : metaData) {
            System.out.print(meta.getName().toStringUtf8() + "\t");
        }
        System.out.println();
        final List<List<Object>> objs = getResult(result);
        for (List<Object> list : objs) {
            for (Object obj : list) {
                System.out.print(obj + "\t");
            }
            System.out.println();
        }
        System.out.println("" + result.getRowsAffected() + " rows affected.");
    }

    public void playground() throws Exception {
        try (XConnection conn = getConn()) {
            conn.setStreamMode(true);
            final XResult result = conn.execQuery("select 1");
            show(result);
        }
    }
}