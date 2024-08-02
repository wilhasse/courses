const std = @import("std");


// show that function is evaluated at compile time
pub fn main() void {
   
    // data comptime
    var myData: [10]u32 = undefined;

    for (0..myData.len) |i| {
        myData[i] = @intCast(i);
    }

    for (myData) |val| {
        std.debug.print("i {d}\n", .{val});
    }
}
