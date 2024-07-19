# Course

Link: https://codeberg.org/ziglings/exercises  

Learn the ⚡Zig programming language by fixing tiny broken programs.

## Installation

```bash
git clone https://ziglings.org
cd ziglings.org
zig build
```

## 001_hello.zig

```zig
const std = @import("std");

fn main() void {
    std.debug.print("Hello world!\n", .{});
}
```

## 002_std.zig

```zig
const foo = @import("std");

pub fn main() void {
    foo.debug.print("Standard Library.\n", .{});
}
```

## 003_assignment.zig

```zig
    var n: u8 = 50;
    n = n + 5;

    const pi: u32 = 314159;

    const negative_eleven: i8 = -11;
```
