import java.util.LinkedHashMap;
import java.util.Map;

public class MainLauncher {
    private static final Map<String, CommandInfo> COMMANDS = new LinkedHashMap<>();  // Changed to LinkedHashMap to maintain order

    static {
        COMMANDS.put("parsesql", new CommandInfo(ParseSQL.class, "Run SQL parsing tests"));
        COMMANDS.put("query", new CommandInfo(SimpleDbQueryApp.class, "Run database query tests"));
        COMMANDS.put("server", new CommandInfo(SimpleServer.class, "Run simple server"));
    }

    private static class CommandInfo {
        final Class<?> classToRun;
        final String description;

        CommandInfo(Class<?> classToRun, String description) {
            this.classToRun = classToRun;
            this.description = description;
        }
    }

    private static void printHelp() {
        System.out.println("Please provide a command number or name:");
        int index = 1;
        for (Map.Entry<String, CommandInfo> entry : COMMANDS.entrySet()) {
            System.out.printf("  %d) %-12s - %s%n", index++, entry.getKey(), entry.getValue().description);
        }
    }

    private static CommandInfo getCommandInfo(String arg) {
        // Try to parse as number first
        try {
            int index = Integer.parseInt(arg);
            if (index > 0 && index <= COMMANDS.size()) {
                return COMMANDS.values().toArray(new CommandInfo[0])[index - 1];
            }
            System.out.println("Invalid number. Please choose between 1 and " + COMMANDS.size());
            return null;
        } catch (NumberFormatException e) {
            // If not a number, try as command name
            return COMMANDS.get(arg.toLowerCase());
        }
    }

    public static void main(String[] args) {
        if (args.length == 0) {
            printHelp();
            return;
        }

        String command = args[0];
        CommandInfo info = getCommandInfo(command);

        if (info == null) {
            System.out.println("Unknown command: " + command);
            printHelp();
            return;
        }

        String[] remainingArgs = new String[args.length - 1];
        System.arraycopy(args, 1, remainingArgs, 0, args.length - 1);

        try {
            info.classToRun.getMethod("main", String[].class)
                    .invoke(null, (Object) remainingArgs);
        } catch (Exception e) {
            System.err.println("Error running command: " + command);
            e.printStackTrace();
        }
    }
}