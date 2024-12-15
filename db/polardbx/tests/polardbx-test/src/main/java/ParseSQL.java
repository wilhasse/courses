import com.alibaba.polardbx.optimizer.OptimizerContext;
import com.alibaba.polardbx.optimizer.context.ExecutionContext;
import com.alibaba.polardbx.optimizer.core.planner.SqlConverter;
import com.alibaba.polardbx.optimizer.parse.FastsqlParser;
import com.alibaba.polardbx.optimizer.config.table.TableMeta;
import com.alibaba.polardbx.optimizer.config.table.ColumnMeta;
import com.alibaba.polardbx.optimizer.config.table.Field;
import com.alibaba.polardbx.optimizer.config.table.statistic.StatisticManager;
import com.alibaba.polardbx.gms.metadb.table.TableStatus;
import com.alibaba.polardbx.config.ConfigDataMode;
import com.alibaba.polardbx.common.utils.InstanceRole;
import com.alibaba.polardbx.optimizer.PlannerContext;

import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlNodeList;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeSystem;
import org.apache.calcite.sql.type.BasicSqlType;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.externalize.RelWriterImpl;
import java.io.PrintWriter;

import java.util.*;

public class ParseSQL {
    public static void main(String[] args) {
        // SQL
        if (args.length < 1) {
            System.err.println("Usage: java polardbx-test 3 <sql_query>");
            return;
        }
        String sql = args[0];

        // Configure system
        ConfigDataMode.setMode(ConfigDataMode.Mode.MOCK);
        ConfigDataMode.setInstanceRole(InstanceRole.FAST_MOCK);

        // Use a custom schema name instead of 'mysql'
        String schemaName = "teste";

        // Set up execution context
        ExecutionContext context = new ExecutionContext();
        context.setSchemaName(schemaName);

        // Create schema manager and add tables
        TestSchemaManager schemaManager = new TestSchemaManager(schemaName);
        schemaManager.init();

        // Add table definitions
        addTableMetadata(schemaManager);

        // Set up OptimizerContext
        OptimizerContext optimizerContext = new OptimizerContext(schemaName);
        optimizerContext.setSchemaManager(schemaManager);
        optimizerContext.setFinishInit(true);
        OptimizerContext.loadContext(optimizerContext);

        // Create necessary managers
        StatisticManager.setExecutor(null);

        try {
            // Parse SQL to AST
            FastsqlParser parser = new FastsqlParser();
            SqlNodeList astList = parser.parse(sql, context);
            SqlNode ast = astList.get(0);
            System.out.println("Parsed AST: " + ast);

            // Create SQL converter
            SqlConverter converter = SqlConverter.getInstance(context.getSchemaName(), context);

            // Validate SQL
            SqlNode validatedNode = converter.validate(ast);
            System.out.println("Validated SQL: " + validatedNode);

            // Create planner context
            PlannerContext plannerContext = new PlannerContext(context);

            // Convert to RelNode and get plan
            RelNode relNode = converter.toRel(validatedNode, plannerContext);

            // Create a RelWriter to explain the plan
            System.out.println("\nExecution Plan:");
            PrintWriter pw = new PrintWriter(System.out);
            RelWriterImpl relWriter = new RelWriterImpl(pw);
            relNode.explain(relWriter);
            pw.flush();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void addTableMetadata(TestSchemaManager schemaManager) {
        List<ColumnMeta> columns = new ArrayList<>();

        // Add Host column
        RelDataType varcharType = new BasicSqlType(RelDataTypeSystem.DEFAULT, SqlTypeName.VARCHAR);
        Field hostField = new Field("user", "Host", varcharType);
        columns.add(new ColumnMeta("user", "Host", null, hostField));

        // Add User column
        Field userField = new Field("user", "User", varcharType);
        columns.add(new ColumnMeta("user", "User", null, userField));

        // Add Select_priv column (using CHAR(1) since it's typically 'Y'/'N')
        RelDataType charType = new BasicSqlType(RelDataTypeSystem.DEFAULT, SqlTypeName.CHAR);
        Field selectPrivField = new Field("user", "Select_priv", charType);
        columns.add(new ColumnMeta("user", "Select_priv", null, selectPrivField));

        // Create table metadata
        TableMeta tableMeta = new TableMeta(
                schemaManager.getSchemaName(), // schema name
                "user",                        // table name
                columns,                       // columns
                null,                         // primary key
                new ArrayList<>(),            // indexes
                true,                         // is public
                TableStatus.PUBLIC,           // status
                1L,                          // version
                0L                           // timestamp
        );

        schemaManager.putTable("user", tableMeta);
    }
}