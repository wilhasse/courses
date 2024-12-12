import com.alibaba.polardbx.common.model.lifecycle.Lifecycle;
import com.alibaba.polardbx.gms.metadb.table.IndexStatus;
import com.alibaba.polardbx.optimizer.config.table.GsiMetaManager.GsiMetaBean;
import com.alibaba.polardbx.optimizer.config.table.SchemaManager;
import com.alibaba.polardbx.optimizer.config.table.TableMeta;
import java.util.Collection;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;

public class TestSchemaManager implements SchemaManager {
    private final Map<String, TableMeta> tables = new HashMap<>();
    private final String schemaName;
    private boolean inited = false;

    public TestSchemaManager(String schemaName) {
        this.schemaName = schemaName;
    }

    @Override
    public TableMeta getTable(String tableName) {
        return tables.get(tableName);
    }

    @Override
    public void putTable(String tableName, TableMeta tableMeta) {
        tables.put(tableName, tableMeta);
    }

    @Override
    public Collection<TableMeta> getAllTables() {
        return tables.values();
    }

    @Override
    public String getSchemaName() {
        return schemaName;
    }

    @Override
    public void init() {
        inited = true;
    }

    @Override
    public void destroy() {
        tables.clear();
        inited = false;
    }

    @Override
    public boolean isInited() {
        return inited;
    }

    @Override
    public void invalidate(String tableName) {
        // No-op for test implementation
    }

    @Override
    public void invalidateAll() {
        // No-op for test implementation
    }

    @Override
    public void reload(String tableName) {
        // No-op for test implementation
    }

    @Override
    public GsiMetaBean getGsi(String primaryOrIndexTableName, EnumSet<IndexStatus> statusSet) {
        // No GSI support in test implementation
        return null;
    }
}