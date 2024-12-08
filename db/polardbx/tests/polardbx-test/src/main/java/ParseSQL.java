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

import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

public class ParseSQL {

    public static void main(String[] args) {
        String sql = "SELECT * FROM users WHERE id = 1";
        try {
            // Configure system
            configureSystem();

            parseSqlAndGeneratePlan(sql);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void configureSystem() {
        // Disable metadata database requirement
        MetaDbInstConfigManager.setConfigFromMetaDb(false);

        // Set instance role to MASTER
        ConfigDataMode.setInstanceRole(InstanceRole.MASTER);

        // Initialize system properties
        Properties props = new Properties();
        props.setProperty(ConnectionProperties.MAX_PHYSICAL_PARTITION_COUNT, "8192");
        props.setProperty(ConnectionProperties.PLAN_CACHE, "false");

        // Set these properties in the system
        for (String name : props.stringPropertyNames()) {
            System.setProperty(name, props.getProperty(name));
        }
    }

    public static void parseSqlAndGeneratePlan(String sql) throws Exception {
        // Step 1: Initialize the execution context
        ExecutionContext executionContext = new ExecutionContext();
        executionContext.setTraceId("test-trace");

        // Initialize Parameters
        Parameters params = new Parameters();
        Map<Integer, ParameterContext> currentParams = new HashMap<>();
        params.setParams(currentParams);
        executionContext.setParams(params);

        // Initialize ParamManager with properties
        Properties connProps = new Properties();
        connProps.setProperty(ConnectionProperties.PLAN_CACHE, "false");
        ParamManager paramManager = new ParamManager(connProps);
        executionContext.setParamManager(paramManager);

        // Initialize other required context elements
        executionContext.setSchemaName("mysql");  // Set default schema
        executionContext.setUserDefVariables(new HashMap<>());  // User variables
        executionContext.setServerVariables(new HashMap<>());   // Server variables
        executionContext.setExtraCmds(new HashMap<>());        // Extra commands

        // Set extra commands
        Map<String, Object> hints = new HashMap<>();
        hints.put(ConnectionProperties.PLAN_CACHE, false);
        executionContext.setExtraCmds(hints);

        // Step 2: Create a planner instance
        Planner planner = Planner.getInstance();

        try {
            // Step 3: Plan the SQL using ByteString
            ExecutionPlan executionPlan = planner.plan(ByteString.from(sql), executionContext);

            // Step 4: Print plan information
            System.out.println("Execution Plan:");
            System.out.println(executionPlan.toString());

        } catch (Exception e) {
            System.err.println("Error during plan generation: " + e.getMessage());
            throw e;
        }
    }
}