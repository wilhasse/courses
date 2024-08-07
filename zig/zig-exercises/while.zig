const std = @import("std");

pub fn main() void {
    std.debug.print("Hello world!\n", .{});

    var n: u32 = 1;
    while (n < 100) : (n *= 2) {
        std.debug.print("{}\n", .{n});
    }
}
