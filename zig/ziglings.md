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

## 004_arrays.zig

```zig
const some_primes = [_]u8{ 1, 3, 5, 7, 11, 13, 17, 19 };
const fourth = some_primes[3];
const length = some_primes.len;
```

## 005_arrays2.zig

```zig
const le = [_]u8{ 1, 3 };
const et = [_]u8{ 3, 7 };
const leet = le ++ et;

// It should result in: 1 0 0 1 1 0 0 1 1 0 0 1
const bit_pattern = [_]u8{ 1 , 0 , 0 , 1 } ** 3;
```

## 006_strings.zig

```zig
const laugh = "ha " ** 3;

const major = "Major";
const tom = "Tom";
const major_tom = major ++ " " ++ tom;
```

## 007_strings2.zig

```zig
pub fn main() void {
    const lyrics =
        \\Ziggy played guitar
        \\Jamming good with Andrew Kelley
        \\And the Spiders from Mars
    ;

    std.debug.print("{s}\n", .{lyrics});
}
```

## 008_quiz.zig

```zig
var x: usize = 1;
var lang: [3]u8 = undefined;
```

## 009_if.zig

```zig
const foo = true;
if (foo) {
    // We want our program to print this message!
    std.debug.print("Foo is 1!\n", .{});
} else {
    std.debug.print("Foo is not 1!\n", .{});
}
```

## 010_if2.zig

```zig
const price: u8 = if (discount) 17 else 20;
std.debug.print("With the discount, the price is ${}.\n", .{price});
```

## 011_while.zig

```zig
while (n < 1024) {
     // Print the current number
    std.debug.print("{} ", .{n});

    // Set n to n multiplied by 2
     n *= 2;
}
```

## 012_while2.zig

```zig
while (n < 1000) : (n *= 2) {
    // Print the current number
    std.debug.print("{} ", .{n});
}
```

## 013_while3.zig

```zig
while (n <= 20) : (n += 1) {
    // The '%' symbol is the "modulo" operator and it
    // returns the remainder after division.
    if (n % 3 == 0) continue;
    if (n % 5 == 0) continue;
    std.debug.print("{} ", .{n});
}
```

## 014_while4.zig

```zig
while (true) : (n += 1) {
    if (n == 4) break;
}

// Result: we want n=4
std.debug.print("n={}\n", .{n});
```

## 015_for.zig

```zig
const story = [_]u8{ 'h', 'h', 's', 'n', 'h' };

for (story) | scene| {
    if (scene == 'h') std.debug.print(":-)  ", .{});
    if (scene == 's') std.debug.print(":-(  ", .{});
    if (scene == 'n') std.debug.print(":-|  ", .{});
}
```

## 016_for2.zig

```zig
for (bits, 0..) |bit, i| {
    // Note that we convert the usize i to a u32 with
    // @intCast(), a builtin function just like @import().
    // We'll learn about these properly in a later exercise.
    const i_u32: u32 = @intCast(i);
    const place_value = std.math.pow(u32, 2, i_u32);
    value += place_value * bit;
}
```
