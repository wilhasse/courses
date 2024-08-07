const std = @import("std");

// show that function is evaluated at compile time
pub fn main() void {

    // data comptime
    var myData: [4][]const u8 = undefined;

    myData[0] = "Abacaxi";
    myData[1] = "Laranja";
    myData[2] = "Morango";
    // try dmyData[3] to runtime error
    std.debug.print("{s}\n", .{myData[2]});
}
