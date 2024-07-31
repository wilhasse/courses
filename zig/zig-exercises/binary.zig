const std = @import("std");

pub fn main() void {


    // undefined
    // var v: [10]u8 = undefined;
    // all zero
    var v: [10]u8 = [_]u8{0} ** 10; 
    v[0] = 0b01001000;
    v[1] = 0x65;
    for (v, 0..) | value , index| {

      std.debug.print("[{d}] {d}\n", .{ index, value });
    }
}
