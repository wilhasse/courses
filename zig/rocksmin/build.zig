const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "main",
        .root_source_file = .{ .cwd_relative = "main.zig" },
        .target = target,
        .optimize = optimize,
    });
    
    exe.linkLibC();
    exe.linkSystemLibrary("rocksdb");
    exe.addLibraryPath(.{ .cwd_relative = "./rocksdb" });
    exe.addIncludePath(.{ .cwd_relative = "./rocksdb/include" });

    b.installArtifact(exe);
}
