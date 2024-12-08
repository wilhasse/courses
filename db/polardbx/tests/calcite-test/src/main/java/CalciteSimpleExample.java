import org.apache.calcite.schema.*;
import org.apache.calcite.schema.impl.AbstractTable;
import org.apache.calcite.linq4j.Enumerable;
import org.apache.calcite.linq4j.Linq4j;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.tools.*;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeSystem;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.parser.SqlParseException;
import org.apache.calcite.sql.parser.SqlParser;
import org.apache.calcite.sql.type.BasicSqlType;
import org.apache.calcite.sql.type.SqlTypeFactoryImpl;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.DataContext;
import org.apache.calcite.rel.externalize.RelWriterImpl;
import java.io.PrintWriter;
import java.util.*;

public class CalciteSimpleExample {
    public static class User {
        public final int id;
        public final String name;
        public User(int id, String name) {
            this.id = id;
            this.name = name;
        }
    }
    
    static class SimpleTable extends AbstractTable implements ScannableTable {
        private final RelDataType rowType;
        SimpleTable(RelDataType rowType) {
            this.rowType = rowType;
        }
        @Override
        public RelDataType getRowType(RelDataTypeFactory typeFactory) {
            return rowType;
        }
        @Override
        public Enumerable<Object[]> scan(DataContext root) {
            return Linq4j.asEnumerable(new Object[][] {
                {1, "Alice"},
                {2, "Bob"},
                {3, "Charlie"}
            });
        }
    }
    
    public static void main(String[] args) throws Exception {
        try {
            // Create root schema
            SchemaPlus rootSchema = Frameworks.createRootSchema(true);
            
            // Create type factory
            RelDataTypeFactory typeFactory = new SqlTypeFactoryImpl(RelDataTypeSystem.DEFAULT);

            // Define the table structure
            RelDataType intType = typeFactory.createSqlType(SqlTypeName.INTEGER);
            RelDataType stringType = typeFactory.createSqlType(SqlTypeName.VARCHAR);
            
            RelDataType rowType = typeFactory.builder()
                .add("ID", intType)
                .add("NAME", stringType)
                .build();

            // Add the table to the schema
            rootSchema.add("USERS", new SimpleTable(rowType));
            
            // Make parser case-insensitive
            SqlParser.Config parserConfig = SqlParser.config()
                .withCaseSensitive(false);

            // Create the final config
            FrameworkConfig config = Frameworks.newConfigBuilder()
                .parserConfig(parserConfig)
                .defaultSchema(rootSchema)
                .build();

            // Try different SQL queries
            String[] queries = {
                "SELECT * FROM USERS",
                "SELECT * FROM users WHERE ID = 1",
                "SELECT name FROM USERS WHERE id > 1",
                "SELECT COUNT(*) FROM users"
            };

            // Create a planner for queries
            Planner planner = Frameworks.getPlanner(config);

            for (String sql : queries) {
                System.out.println("\nExecuting query: " + sql);
                try {
                    // Parse and validate
                    SqlNode parse = planner.parse(sql);
                    SqlNode validate = planner.validate(parse);
                    
                    // Convert to rel
                    RelNode rel = planner.rel(validate).project();
                    
                    System.out.println("Execution plan:");
                    rel.explain(new RelWriterImpl(new PrintWriter(System.out, true)));
                } catch (Exception e) {
                    System.out.println("Error executing query: " + e.getMessage());
                }
                
                // Reset planner for next query
                planner.close();
                planner = Frameworks.getPlanner(config);
            }

        } catch (Exception e) {
            System.out.println("Main error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}