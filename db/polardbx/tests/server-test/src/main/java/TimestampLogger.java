import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A simple logger that records timestamps for tasks, then logs total duration.
 */
public class TimestampLogger {
    private static final SimpleDateFormat TIME_FORMAT = new SimpleDateFormat("HH:mm:ss.SSS");
    private static final Map<String, Long> startTimes = new ConcurrentHashMap<>();

    /**
     * Prints a log message with a current timestamp.
     *
     * @param message The message to log
     */
    public static void logWithTime(String message) {
        String timestamp = TIME_FORMAT.format(new Date());
        System.out.println(timestamp + " - " + message);
    }

    /**
     * Records the start time for a task with the given ID.
     *
     * @param taskId A unique string ID for the task
     */
    public static void startTimer(String taskId) {
        startTimes.put(taskId, System.currentTimeMillis());
        logWithTime("Starting task: " + taskId);
    }

    /**
     * Logs a message with the duration since startTimer was called for this taskId.
     *
     * @param taskId  The task ID
     * @param message The message to log
     */
    public static void logWithDuration(String taskId, String message) {
        Long startTime = startTimes.get(taskId);
        if (startTime != null) {
            long duration = System.currentTimeMillis() - startTime;
            logWithTime(message + " (Duration: " + duration + "ms)");
        } else {
            logWithTime(message + " (No start time found for task: " + taskId + ")");
        }
    }
}
