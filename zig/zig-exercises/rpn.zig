const std = @import("std");

pub fn Stack(comptime T: type) type {
    return struct {
        items: std.ArrayList(T),
        const Self = @This();
        pub fn init(allocator: std.mem.Allocator) Self {
            return Self{
                .items = std.ArrayList(T).init(allocator),
            };
        }
        pub fn deinit(self: *Self) void {
            self.items.deinit();
        }
        pub fn push(self: *Self, value: T) !void {
            try self.items.append(value);
        }
        pub fn pop(self: *Self) ?T {
            return if (self.items.items.len == 0) null else self.items.pop();
        }
        pub fn peek(self: *Self) ?T {
            return if (self.items.items.len == 0) null else self.items.items[self.items.items.len - 1];
        }
        pub fn isEmpty(self: *Self) bool {
            return self.items.items.len == 0;
        }
        pub fn size(self: *Self) usize {
            return self.items.items.len;
        }
    };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const IntStack = Stack(i32);
    const StringStack = Stack([]const u8);
    
    var intStack = IntStack.init(allocator);
    defer intStack.deinit();
    
    var stringStack = StringStack.init(allocator);
    defer stringStack.deinit();

    try intStack.push(42);        // This is allowed
    try stringStack.push("Hello"); // This is allowed
    //try intStack.push("World");   // This would cause a compile-time error

    // Testing IntStack
    try intStack.push(42);
    try intStack.push(23);
    try intStack.push(16);

    std.debug.print("IntStack:\n", .{});
    while (intStack.pop()) |value| {
        std.debug.print("Popped: {}\n", .{value});
    }
    std.debug.print("IntStack is empty: {}\n\n", .{intStack.isEmpty()});

    // Testing StringStack
    try stringStack.push("Hello");
    try stringStack.push("World");
    try stringStack.push("Zig");

    std.debug.print("StringStack:\n", .{});
    while (stringStack.pop()) |value| {
        std.debug.print("Popped: {s}\n", .{value});
    }
    std.debug.print("StringStack is empty: {}\n", .{stringStack.isEmpty()});

}
