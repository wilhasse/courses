const std = @import("std");

fn sizeArray(comptime size: usize) bool {
    return comptime size > 10;
}

fn sizeArrayComp(comptime size: usize) bool {
    return comptime size > 10;
}

// show that function is evaluated at compile time
pub fn main() void {
   
    // data comptime
    const myData = comptime [_]u32{1, 2, 3, 4, 5};
   
    // This will be evaluated at compile time
    const isLarge  = comptime sizeArrayComp(myData.len); 

    // This will be evaluated at run time
    const isLargeR = sizeArray(myData.len);

    if (isLarge) {
        std.debug.print("Array is large, special processing was applied at compile time.\n", .{});
    } else {
        std.debug.print("Array is small, standard processing was applied at compile time.\n", .{});
    }

    if (isLargeR) {
        std.debug.print("Runtime: Array is large, special processing was applied at compile time.\n", .{});
    } else {
        std.debug.print("Runtime: Array is small, standard processing was applied at compile time.\n", .{});
    }
}
