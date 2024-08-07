const std = @import("std");

pub fn main() !void {
    // For loop example
    const items = [_]i32{ 1, 2, 3, 4, 5 };
    for (items, 0..) |item, index| {
        std.debug.print("Item {} at index {}\n", .{ item, index });
    }

    // While loop example (C-style for loop)
    var i: usize = 0;
    while (i < 5) : (i += 1) {
        std.debug.print("Counter: {}\n", .{i});
    }

    // Labeled break example
    outer: for (items) |item| {
        if (item == 3) {
            std.debug.print("Found 3, breaking outer loop\n", .{});
            break :outer;
        }
    }

    // Loop as an expression
    const sum = blk: {
        var total: i32 = 0;
        for (items) |item| {
            total += item;
        }
        break :blk total;
    };
    std.debug.print("Sum of items: {}\n", .{sum});
}
