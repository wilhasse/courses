const std = @import("std");

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();

    // Basic types
    const number: i32 = 42;
    const pi: f64 = 3.14159;
    const letter: u8 = 'A';
    const text = "Hello";

    try stdout.print("Default:     {}\n", .{number});
    try stdout.print("Decimal:     {d}\n", .{number});
    try stdout.print("Hexadecimal: 0x{X:0>4}\n", .{number});
    try stdout.print("Binary:      0b{b:0>8}\n", .{number});
    
    try stdout.print("Float:       {d:.2}\n", .{pi});
    try stdout.print("Scientific:  {e}\n", .{pi});
    
    try stdout.print("Character:   {c}\n", .{letter});
    try stdout.print("String:      {s}\n", .{text});

    // Width and alignment
    try stdout.print("Left align:  {:<10}\n", .{number});
    try stdout.print("Center:      {:^10}\n", .{number});
    try stdout.print("Right align: {:>10}\n", .{number});
    
    // Fill character
    try stdout.print("Fill zeros:  {:0>5}\n", .{number});
    
    // Combining options
    try stdout.print("Complex:     {:->10.2}\n", .{pi});
}
