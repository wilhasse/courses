// Zig version using comptime
const std = @import("std");

const SnekObjectKind = enum {
    Integer,
    Float,
    Bool,
};

const SnekInt = struct {
    name: []const u8,
    value: i32,
};

const SnekFloat = struct {
    name: []const u8,
    value: f32,
};

const SnekBool = struct {
    name: []const u8,
    value: bool,
};

// Generic function using comptime type information
fn snekZeroOut(ptr: anytype) void {
    const T = @TypeOf(ptr);
    const info = @typeInfo(T);
    
    // Ensure we're working with a pointer
    if (info != .Pointer) {
        @compileError("Expected pointer, got " ++ @typeName(T));
    }
    
    const child = info.Pointer.child;
    switch (child) {
        SnekInt => ptr.value = 0,
        SnekFloat => ptr.value = 0.0,
        SnekBool => ptr.value = false,
        else => @compileError("Unsupported type: " ++ @typeName(child)),
    }
}

pub fn main() !void {
    var my_int = SnekInt{ .name = "integer", .value = 42 };
    var my_float = SnekFloat{ .name = "float", .value = 3.14 };
    var my_bool = SnekBool{ .name = "boolean", .value = true };

    // Print initial values
    std.debug.print("Before: int={}, float={d}, bool={}\n", 
        .{my_int.value, my_float.value, my_bool.value});

    // Zero out values
    snekZeroOut(&my_int);
    snekZeroOut(&my_float);
    snekZeroOut(&my_bool);

    // Print final values
    std.debug.print("After: int={}, float={d}, bool={}\n", 
        .{my_int.value, my_float.value, my_bool.value});
}
