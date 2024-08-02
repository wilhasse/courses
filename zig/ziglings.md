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

## 017_quiz2.zig

```zig
pub fn main() void {
    var i: u8 = 1;
    const stop_at: u8 = 16;

    // What kind of loop is this? A 'for' or a 'while'?
    while (i <= stop_at) : (i += 1) {
        if (i % 3 == 0) std.debug.print("Fizz", .{});
        if (i % 5 == 0) std.debug.print("Buzz", .{});
        if (!(i % 3 == 0) and !(i % 5 == 0)) {
            std.debug.print("{}", .{i});
        }
        std.debug.print(", ", .{});
    }
    std.debug.print("\n", .{});
}
```

## 018_functions.zig

```zig
# private function
fn deepThought() u8 {
    return 42; // Number courtesy Douglas Adams
}
```

## 019_functions2.zig

```zig
fn twoToThe(my_number: u32) u32 {
    return std.math.pow(u32, 2, my_number);
}
```

## 020_quiz3.zig

```zig
fn printPowersOfTwo(numbers: [4]u16) void {
    for (numbers) |n| {
        std.debug.print("{} ", .{twoToThe(n)});
    }
}

fn twoToThe(number: u16) u16 {
    var n: u16 = 0;
    var total: u16 = 1;

    while (n < number) : (n += 1) {
        total *= 2;
    }

    return total;
}
```

## 021_errors.zig

```zig
const MyNumberError = error{
    TooBig,
    TooSmall,
    TooFour,
};

pub fn main() void {
    const nums = [_]u8{ 2, 3, 4, 5, 6 };

    for (nums) |n| {
        std.debug.print("{}", .{n});

        const number_error = numberFail(n);

        if (number_error == MyNumberError.TooBig) {
            std.debug.print(">4. ", .{});
        }
        if (number_error == MyNumberError.TooSmall) {
            std.debug.print("<4. ", .{});
        }
        if (number_error == MyNumberError.TooFour) {
            std.debug.print("=4. ", .{});
        }
    }

    std.debug.print("\n", .{});
}

fn numberFail(n: u8) MyNumberError {
    if (n > 4) return MyNumberError.TooBig;
    if (n < 4) return MyNumberError.TooSmall; // <---- this one is free!
    return MyNumberError.TooFour;
}
```

## 022_errors2.zig

```zig
const std = @import("std");

const MyNumberError = error{TooSmall};

pub fn main() void {
    var my_number: MyNumberError!u16 = 5;

    // Looks like my_number will need to either store a number OR
    // an error. Can you set the type correctly above?
    my_number = MyNumberError.TooSmall;

    std.debug.print("I compiled!\n", .{});
}
```

## 023_errors3.zig

```zig
const MyNumberError = error{TooSmall};

pub fn main() void {
    const a: u32 = addTwenty(44) catch 22;
    const b: u32 = addTwenty(4) catch 22;

    std.debug.print("a={}, b={}\n", .{ a, b });
}

// Please provide the return type from this function.
// Hint: it'll be an error union.
fn addTwenty(n: u32) MyNumberError!u32 {
    if (n < 5) {
        return MyNumberError.TooSmall;
    } else {
        return n + 20;
    }
}
```

## 024_errors4.zig

```zig
fn fixTooSmall(n: u32) MyNumberError!u32 {
    // Oh dear, this is missing a lot! But don't worry, it's nearly
    // identical to fixTooBig() above.
    //
    // If we get a TooSmall error, we should return 10.
    // If we get any other error, we should return that error.
    // Otherwise, we return the u32 number.
    return detectProblems(n) catch |err| {
        if (err == MyNumberError.TooSmall) {
            return 10;
        }

        return err;
    };
}
```

## 025_errors5.zig

```zig
fn addFive(n: u32) MyNumberError!u32 {
    // This function needs to return any error which might come back from detect().
    // Please use a "try" statement rather than a "catch".
    //
    const x = try detect(n);

    return x + 5;
}
```

## 026_hello2.zig

```zig
pub fn main() !void {
    // We get a Writer for Standard Out so we can print() to it.
    const stdout = std.io.getStdOut().writer();

    // Unlike std.debug.print(), the Standard Out writer can fail
    // with an error. We don't care _what_ the error is, we want
    // to be able to pass it up as a return value of main().
    //
    // We just learned of a single statement which can accomplish this.
    try stdout.print("Hello world!\n", .{});
}
```

## 027_defer.zig

```zig
pub fn main() void {
    // Without changing anything else, please add a 'defer' statement
    // to this code so that our program prints "One Two\n":
    defer std.debug.print("Two\n", .{});
    std.debug.print("One ", .{});
}
```

## 028_defer2.zig

```zig
fn printAnimal(animal: u8) void {
    std.debug.print("(", .{});

    defer std.debug.print(") ", .{}); // <---- how?!

    if (animal == 'g') {
        std.debug.print("Goat", .{});
        return;
    }
    if (animal == 'c') {
        std.debug.print("Cat", .{});
        return;
    }
    if (animal == 'd') {
        std.debug.print("Dog", .{});
        return;
    }

    std.debug.print("Unknown", .{});
}
```

## 029_errdefer.zig

```zig
fn makeNumber() MyErr!u32 {
    std.debug.print("Getting number...", .{});

    // Please make the "failed" message print ONLY if the makeNumber()
    // function exits with an error:
    errdefer std.debug.print("failed!\n", .{});

    var num = try getNumber(); // <-- This could fail!

    num = try increaseNumber(num); // <-- This could ALSO fail!

    std.debug.print("got {}. ", .{num});

    return num;
}
```

## 030_switch.zig

```zig
pub fn main() void {
    const lang_chars = [_]u8{ 26, 9, 7, 42 };

    for (lang_chars) |c| {
        switch (c) {
            1 => std.debug.print("A", .{}),
            2 => std.debug.print("B", .{}),
            3 => std.debug.print("C", .{}),
            // ... we don't need everything in between ...
            25 => std.debug.print("Y", .{}),
            26 => std.debug.print("Z", .{}),
            // Switch statements must be "exhaustive" (there must be a
            // match for every possible value).  Please add an "else"
            // to this switch to print a question mark "?" when c is
            // not one of the existing matches.
            //
            else => std.debug.print("?", .{}),
        }
    }

    std.debug.print("\n", .{});
}
```

## 031_switch2.zig

```zig
pub fn main() void {
    const lang_chars = [_]u8{ 26, 9, 7, 42 };

    for (lang_chars) |c| {
        const real_char: u8 = switch (c) {
            1 => 'A',
            7 => 'G',
            9 => 'I',
            26 => 'Z',
            // As in the last exercise, please add the 'else' clause
            // and this time, have it return an exclamation mark '!'.
            else => '!',
        };

        std.debug.print("{c}", .{real_char});

        // Note: "{c}" forces print() to display the value as a character.
        // Can you guess what happens if you remove the "c"? Try it!
    }

    std.debug.print("\n", .{});
}
```

## 032_unreachable.zig

```zig
pub fn main() void {
    const operations = [_]u8{ 1, 1, 1, 3, 2, 2 };

    var current_value: u32 = 0;

    for (operations) |op| {
        switch (op) {
            1 => {
                current_value += 1;
            },
            2 => {
                current_value -= 1;
            },
            3 => {
                current_value *= current_value;
            },
            else => unreachable,
        }

        std.debug.print("{} ", .{current_value});
    }

    std.debug.print("\n", .{});
}

```

## 033_iferror.zig

```zig
pub fn main() void {
    const nums = [_]u8{ 2, 3, 4, 5, 6 };

    for (nums) |num| {
        std.debug.print("{}", .{num});

        const n = numberMaybeFail(num);
        if (n) |value| {
            std.debug.print("={}. ", .{value});
        } else |err| switch (err) {
            MyNumberError.TooBig => std.debug.print(">4. ", .{}),
            MyNumberError.TooSmall => std.debug.print("<4. ", .{}),
        }
    }

    std.debug.print("\n", .{});
}
```

## 034_quiz4.zig

```zig
const NumError = error{IllegalNumber};

pub fn main() ! void {
    const stdout = std.io.getStdOut().writer();

    const my_num: u32 = getNumber() catch 42;

    try stdout.print("my_num={}\n", .{my_num});
}

// This function is obviously weird and non-functional. But you will not be changing it for this quiz.
fn getNumber() NumError!u32 {
    if (false) return NumError.IllegalNumber;
    return 42;
}
```
