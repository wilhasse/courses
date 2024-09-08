const std = @import("std");

const NUM_THREADS = 100;
const NUM_WRITES = 10000;
const FILE_PATH = "test_output.txt";

const FileWriteMode = enum {
    safe,
    unsafe,
};

fn writeToFile(thread_id: usize, mode: FileWriteMode, mutex: *std.Thread.Mutex) !void {
    const content = try std.fmt.allocPrint(std.heap.page_allocator, "Thread {d} writing\n", .{thread_id});
    defer std.heap.page_allocator.free(content);

    var i: usize = 0;
    while (i < NUM_WRITES) : (i += 1) {
        switch (mode) {
            .unsafe => {
                const file = try std.fs.cwd().openFile(FILE_PATH, .{ .mode = .write_only });
                defer file.close();

                try file.seekFromEnd(0);
                _ = try file.write(content);
            },
            .safe => {
                mutex.lock();
                defer mutex.unlock();

                const file = try std.fs.cwd().openFile(FILE_PATH, .{ .mode = .write_only });
                defer file.close();

                try file.seekFromEnd(0);
                _ = try file.write(content);
            },
        }
    }
}

pub fn main() !void {

    // Random write to file using multiple threads
    // safe and unsafe
    //
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    // Skip the program name
    _ = args.next();

    const mode_arg = args.next() orelse {
        std.debug.print("Usage: {s} <safe|unsafe>\n", .{std.os.argv[0]});
        return;
    };

    const mode: FileWriteMode = if (std.mem.eql(u8, mode_arg, "safe"))
        .safe
    else if (std.mem.eql(u8, mode_arg, "unsafe"))
        .unsafe
    else {
        std.debug.print("Invalid mode. Use 'safe' or 'unsafe'.\n", .{});
        return;
    };

    // Create the file
    {
        const file = try std.fs.cwd().createFile(FILE_PATH, .{});
        file.close();
    }

    var threads: [NUM_THREADS]std.Thread = undefined;
    var mutex = std.Thread.Mutex{};

    const start_time = std.time.milliTimestamp();

    // Spawn threads
    for (0..NUM_THREADS) |i| {
        threads[i] = try std.Thread.spawn(.{}, writeToFile, .{ i, mode, &mutex });
    }

    // Wait for all threads to complete
    for (threads) |thread| {
        thread.join();
    }

    const end_time = std.time.milliTimestamp();
    const elapsed_time = end_time - start_time;

    std.debug.print("Test completed in {s} mode.\n", .{@tagName(mode)});
    std.debug.print("Time elapsed: {} milliseconds\n", .{elapsed_time});
    std.debug.print("Check {s} for results.\n", .{FILE_PATH});
}
