const std = @import("std");

const Error = error{
    TooLow,
};

fn mayFail(i: u32) !u32 {

    if (i < 10) {
      return error.TooLow;
    }

    return i;
}

pub fn main() void {

    // try 2 or 20
    // you will get different results
    const result = mayFail(20);
    if (result) |value| {
        // If there is no error, this block is executed.
        std.debug.print("OK: {}\n", .{value});
    } else |err| {
        // If there is an error, this block is executed.
        std.debug.print("Error: {}\n", .{err});
    }
}
