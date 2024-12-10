import com.alibaba.polardbx.optimizer.core.planner.Planner;
import com.alibaba.polardbx.optimizer.context.ExecutionContext;
import com.alibaba.polardbx.optimizer.core.planner.ExecutionPlan;
import com.alibaba.polardbx.druid.sql.parser.ByteString;
import com.alibaba.polardbx.common.jdbc.Parameters;
import com.alibaba.polardbx.common.jdbc.ParameterContext;
import com.alibaba.polardbx.common.properties.ConnectionProperties;
import com.alibaba.polardbx.common.properties.ParamManager;
import com.alibaba.polardbx.gms.config.impl.MetaDbInstConfigManager;
import com.alibaba.polardbx.config.ConfigDataMode;
import com.alibaba.polardbx.common.utils.InstanceRole;
import com.alibaba.polardbx.optimizer.OptimizerContext;
import com.alibaba.polardbx.optimizer.config.table.TableMeta;
import com.alibaba.polardbx.optimizer.config.table.IndexMeta;
import com.alibaba.polardbx.optimizer.config.table.ColumnMeta;
import com.alibaba.polardbx.optimizer.config.table.Field;
import com.alibaba.polardbx.optimizer.config.table.IndexType;
import com.alibaba.polardbx.optimizer.config.table.Relationship;
import com.alibaba.polardbx.gms.metadb.table.TableStatus;
import com.alibaba.polardbx.gms.metadb.table.IndexStatus;
import com.alibaba.polardbx.optimizer.config.table.GsiMetaManager.GsiMetaBean;
import com.alibaba.polardbx.common.model.Matrix;
import com.alibaba.polardbx.optimizer.config.table.SchemaManager;

import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.sql.type.BasicSqlType;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.rel.type.RelDataTypeSystem;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

public class ParseSQL {
    public static void main(String[] args) {
        String sql = "SELECT * FROM tables_priv WHERE id = 1";
        try {
            // Configure system and schema
            configureSystem();
            setupTestSchema();
            parseSqlAndGeneratePlan(sql);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void configureSystem() {
        MetaDbInstConfigManager.setConfigFromMetaDb(false);
        ConfigDataMode.setInstanceRole(InstanceRole.MASTER);

        Properties props = new Properties();
        props.setProperty(ConnectionProperties.MAX_PHYSICAL_PARTITION_COUNT, "8192");
        props.setProperty(ConnectionProperties.PLAN_CACHE, "false");

        for (String name : props.stringPropertyNames()) {
            System.setProperty(name, props.getProperty(name));
        }
    }

    private static void setupTestSchema() {
        // Create schema name
        final String schemaName = "mysql";

        // Create test schema manager with required system tables
        TestSchemaManager schemaManager = new TestSchemaManager(schemaName);
        schemaManager.init();

        // Add users table
        TableMeta usersMeta = createUsersTable(schemaName);
        schemaManager.putTable("users", usersMeta);

        // Add required system tables
        createSystemTables(schemaManager, schemaName);

        // Initialize OptimizerContext
        OptimizerContext optimizerContext = new OptimizerContext(schemaName);
        optimizerContext.setSchemaManager(schemaManager);

        // Setup execution context
        ExecutionContext executionContext = new ExecutionContext();
        executionContext.setSchemaName(schemaName);
        Map<String, SchemaManager> schemaManagers = new HashMap<>();
        schemaManagers.put(schemaName, schemaManager);
        executionContext.setSchemaManagers(schemaManagers);

        // Set up matrix in optimizer context
        Matrix matrix = new Matrix();
        matrix.setName(schemaName);
        optimizerContext.setMatrix(matrix);

        // Finish initialization and load context
        optimizerContext.setFinishInit(true);
        OptimizerContext.loadContext(optimizerContext);
    }

    private static TableMeta createUsersTable(String schemaName) {
        List<ColumnMeta> columns = new ArrayList<>();

        // Create Calcite types
        RelDataType bigintType = new BasicSqlType(RelDataTypeSystem.DEFAULT, SqlTypeName.BIGINT);
        RelDataType varcharType = new BasicSqlType(RelDataTypeSystem.DEFAULT, SqlTypeName.VARCHAR);

        // Create ID column
        Field idField = new Field("users", "id", bigintType, false, true);
        ColumnMeta idColumn = new ColumnMeta("users", "id", null, idField);
        columns.add(idColumn);

        // Create name column
        Field nameField = new Field("users", "name", varcharType);
        ColumnMeta nameColumn = new ColumnMeta("users", "name", null, nameField);
        columns.add(nameColumn);

        // Create primary key
        List<ColumnMeta> pkColumns = new ArrayList<>();
        pkColumns.add(idColumn);

        // Create primary index
        IndexMeta primaryIndex = new IndexMeta(
                "users",
                pkColumns,
                new ArrayList<>(),
                IndexType.BTREE,
                Relationship.ONE_TO_ONE,
                true,
                true,
                true,
                "PRIMARY"
        );

        return new TableMeta(
                schemaName,
                "users",
                columns,
                primaryIndex,
                new ArrayList<>(),
                true,
                TableStatus.PUBLIC,
                1L,
                0L
        );
    }

    private static void createSystemTables(TestSchemaManager schemaManager, String schemaName) {
        // List of required system tables
        String[] systemTables = {
                "tables_priv",
                "user",
                "db",
                "help_topic",
                "help_category",
                "help_relation",
                "help_keyword",
                "columns_priv",
                "procs_priv",
                "proxies_priv"
        };

        // Create a basic table structure for each system table
        for (String tableName : systemTables) {
            List<ColumnMeta> columns = new ArrayList<>();

            // Create a simple ID column for each system table
            RelDataType idType = new BasicSqlType(RelDataTypeSystem.DEFAULT, SqlTypeName.BIGINT);
            Field idField = new Field(tableName, "id", idType, false, true);
            ColumnMeta idColumn = new ColumnMeta(tableName, "id", null, idField);
            columns.add(idColumn);

            // Create primary key
            List<ColumnMeta> pkColumns = new ArrayList<>();
            pkColumns.add(idColumn);

            // Create primary index
            IndexMeta primaryIndex = new IndexMeta(
                    tableName,
                    pkColumns,
                    new ArrayList<>(),
                    IndexType.BTREE,
                    Relationship.ONE_TO_ONE,
                    true,
                    true,
                    true,
                    "PRIMARY"
            );

            TableMeta tableMeta = new TableMeta(
                    schemaName,
                    tableName,
                    columns,
                    primaryIndex,
                    new ArrayList<>(),
                    true,
                    TableStatus.PUBLIC,
                    1L,
                    0L
            );

            schemaManager.putTable(tableName, tableMeta);
        }
    }

    private static class TestSchemaManager implements SchemaManager {
        private final Map<String, TableMeta> tables = new HashMap<>();
        private final String schemaName;
        private boolean inited = false;
        private final Set<String> systemTables;

        public TestSchemaManager(String schemaName) {
            this.schemaName = schemaName;
            this.systemTables = new HashSet<>(Arrays.asList(
                    "tables_priv", "user", "db", "help_topic", "help_category",
                    "help_relation", "help_keyword", "columns_priv", "procs_priv", "proxies_priv"
            ));
        }

        @Override
        public TableMeta getTable(String tableName) {
            if (tableName == null) {
                return null;
            }
            String lowercaseTableName = tableName.toLowerCase();
            if (systemTables.contains(lowercaseTableName)) {
                return tables.get(lowercaseTableName);
            }
            return tables.get(tableName);
        }

        @Override
        public TableMeta getTableWithNull(String tableName) {
            if (tableName == null) {
                return null;
            }
            return getTable(tableName);
        }

        @Override
        public void putTable(String tableName, TableMeta tableMeta) {
            if (systemTables.contains(tableName.toLowerCase())) {
                tables.put(tableName.toLowerCase(), tableMeta);
            } else {
                tables.put(tableName, tableMeta);
            }
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
        public void invalidate(String tableName) {}

        @Override
        public void invalidateAll() {}

        @Override
        public void reload(String tableName) {}

        @Override
        public GsiMetaBean getGsi(String primaryOrIndexTableName, EnumSet<IndexStatus> statusSet) {
            return null;
        }
    }

    public static void parseSqlAndGeneratePlan(String sql) throws Exception {
        // Initialize execution context with schema information
        ExecutionContext executionContext = new ExecutionContext();
        executionContext.setTraceId("test-trace");
        executionContext.setSchemaName("mysql");

        // Get schema manager from optimizer context
        SchemaManager schemaManager = OptimizerContext.getContext("mysql").getLatestSchemaManager();

        // Set up schema managers map
        Map<String, SchemaManager> schemaManagers = new HashMap<>();
        schemaManagers.put("mysql", schemaManager);
        executionContext.setSchemaManagers(schemaManagers);

        // Setup parameters
        Parameters params = new Parameters();
        Map<Integer, ParameterContext> currentParams = new HashMap<>();
        params.setParams(currentParams);
        executionContext.setParams(params);

        // Setup connection properties
        Properties connProps = new Properties();
        connProps.setProperty(ConnectionProperties.PLAN_CACHE, "false");
        ParamManager paramManager = new ParamManager(connProps);
        executionContext.setParamManager(paramManager);

        // Initialize variables and commands
        executionContext.setUserDefVariables(new HashMap<>());
        executionContext.setServerVariables(new HashMap<>());

        Map<String, Object> hints = new HashMap<>();
        hints.put(ConnectionProperties.PLAN_CACHE, false);
        executionContext.setExtraCmds(hints);

        // Create planner and generate plan
        Planner planner = Planner.getInstance();
        try {
            ExecutionPlan executionPlan = planner.plan(ByteString.from(sql), executionContext);
            System.out.println("Execution Plan:");
            System.out.println(executionPlan.toString());
        } catch (Exception e) {
            System.err.println("Error during plan generation: " + e.getMessage());
            throw e;
        }
    }
}