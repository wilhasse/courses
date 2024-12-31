import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class TimestampLogger {
    private static final SimpleDateFormat TIME_FORMAT = new SimpleDateFormat("HH:mm:ss.SSS");
    private static final Map<String, Long> startTimes = new ConcurrentHashMap<>();

    public static void logWithTime(String message) {
        String timestamp = TIME_FORMAT.format(new Date());
        System.out.println(timestamp + " - " + message);
    }

    public static void startTimer(String taskId) {
        startTimes.put(taskId, System.currentTimeMillis());
        logWithTime("Starting task: " + taskId);
    }

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