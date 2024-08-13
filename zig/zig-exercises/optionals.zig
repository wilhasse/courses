const std = @import("std");

// show that function is evaluated at compile time
pub fn main() void {
    const foo: ?u32 = null;

    //foo = 3;
    const bar = foo orelse 2;

    // try dmyData[3] to runtime error
    std.debug.print("{?}\n", .{foo});
    std.debug.print("{?}\n", .{bar});
}
