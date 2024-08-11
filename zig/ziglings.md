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

## 035_enum.zig

```zig
// Please complete the enum!
const Ops = enum { inc,dec,pow };

pub fn main() void {
    const operations = [_]Ops{
        Ops.inc,
        Ops.inc,
        Ops.inc,
        Ops.pow,
        Ops.dec,
        Ops.dec,
    };
}
```

## 036_enums2.zig

```zig
const Color = enum(u32) {
    red = 0xff0000,
    green = 0x00ff00,
    blue = 0x0000ff,
};

pub fn main() void {
    std.debug.print(
        \\<p>
        \\  <span style="color: #{x:0>6}">Red</span>
        \\  <span style="color: #{x:0>6}">Green</span>
        \\  <span style="color: #{x:0>6}">Blue</span>
        \\</p>
        \\
    , .{
        @intFromEnum(Color.red),
        @intFromEnum(Color.green),
        @intFromEnum(Color.blue),
    });
}
```

## 037_structs.zig

```zig
// Please add a new property to this struct called "health" and make
// it a u8 integer type.
const Character = struct {
    role: Role,
    gold: u32,
    experience: u32,
    health: u8,
};

pub fn main() void {
    // Please initialize Glorp with 100 health.
    var glorp_the_wise = Character{
        .role = Role.wizard,
        .gold = 20,
        .experience = 10,
        .health = 100,
    };

    // Glorp gains some gold.
    glorp_the_wise.gold += 5;

    // Ouch! Glorp takes a punch!
    glorp_the_wise.health -= 10;

    std.debug.print("Your wizard has {} health and {} gold.\n", .{
        glorp_the_wise.health,
        glorp_the_wise.gold,
    });
}
```

## 038_structs.zig

```zig
pub fn main() void {
    var chars: [2]Character = undefined;

    chars[0] = Character{
        .role = Role.wizard,
        .gold = 20,
        .health = 100,
        .experience = 10,
    };

    chars[1] = Character{
        .role = Role.wizard,
        .gold = 10,
        .health = 100,
        .experience = 20,
    };

    // Printing all RPG characters in a loop:
    for (chars, 0..) |c, num| {
        std.debug.print("Character {} - G:{} H:{} XP:{}\n", .{
            num + 1, c.gold, c.health, c.experience,
        });
    }
}
```

## 039_pointers.zig

```zig
pub fn main() void {
    var num1: u8 = 5;
    const num1_pointer: *u8 = &num1;

    var num2: u8 = undefined;

    // Please make num2 equal 5 using num1_pointer!
    // (See the "cheatsheet" above for ideas.)
    num2 = num1_pointer.*;

    std.debug.print("num1: {}, num2: {}\n", .{ num1, num2 });
}
```

## 040_pointers2.zig

```zig
pub fn main() void {
    var a: u8 = 12;
    const b: *u8 = &a; // fix this!

    std.debug.print("a: {}, b: {}\n", .{ a, b.* });
}
```

## 041_pointers3.zig

```zig
pub fn main() void {
    var foo: u8 = 5;
    var bar: u8 = 10;

    // Please define pointer "p" so that it can point to EITHER foo or
    // bar AND change the value it points to!
    var p: *u8 = undefined;

    p = &foo;
    p.* += 1;
    p = &bar;
    p.* += 1;
    std.debug.print("foo={}, bar={}\n", .{ foo, bar });
```

## 042_pointers4.zig

```zig
// This function should take a reference to a u8 value and set it
// to 5.
fn makeFive(x: *u8) void {
     x.* = 5; // fix me!
}
```

## 043_pointers5.zig

```zig
    var glorp = Character{ // Glorp!
        .class = Class.wizard,
        .gold = 10,
        .experience = 20,
        .mentor = &mighty_krodor, // Glorp's mentor is the Mighty Krodor
    };

    // FIX ME!
    // Please pass Glorp to printCharacter():
    printCharacter(&glorp);
```

## 044_quiz5.zig

```zig
pub fn main() void {
    var elephantA = Elephant{ .letter = 'A' };
    var elephantB = Elephant{ .letter = 'B' };
    var elephantC = Elephant{ .letter = 'C' };

    // Link the elephants so that each tail "points" to the next elephant.
    // They make a circle: A->B->C->A...
    elephantA.tail = &elephantB;
    elephantB.tail = &elephantC;
    elephantC.tail = &elephantA;

    visitElephants(&elephantA);

    std.debug.print("\n", .{});
}
```

## 045_optionals.zig

```zig
pub fn main() void {
    const result = deepThought();

    // Please threaten the result so that answer is either the
    // integer value from deepThought() OR the number 42:
    const answer: u8 = result orelse 42;

    std.debug.print("The Ultimate Answer: {}.\n", .{answer});
}
```

## 046_optionals2.zig

```zig
// This function visits all elephants once, starting with the
// first elephant and following the tails to the next elephant.
fn visitElephants(first_elephant: *Elephant) void {
    var e = first_elephant;

    while (!e.visited) {
        std.debug.print("Elephant {u}. ", .{e.letter});
        e.visited = true;

        // We should stop once we encounter a tail that
        // does NOT point to another element. What can
        // we put here to make that happen?

        // HINT: We want something similar to what `.?` does,
        // but instead of ending the program, we want to exit the loop...
        e = e.tail orelse break;
    }
}
```

## 047_methods.zig

```zig
// Look at this hideous Alien struct. Know your enemy!
const Alien = struct {
    health: u8,

    // We hate this method:
    pub fn hatch(strength: u8) Alien {
        return Alien{
            .health = strength * 5,
        };
    }
};

// Your trusty weapon. Zap those aliens!
const HeatRay = struct {
    damage: u8,

    // We love this method:
    pub fn zap(self: HeatRay, alien: *Alien) void {
        alien.health -= if (self.damage >= alien.health) alien.health else self.damage;
    }
};

pub fn main() void {
    // Look at all of these aliens of various strengths!
    var aliens = [_]Alien{
        Alien.hatch(2),
        Alien.hatch(1),
    };

    var aliens_alive = aliens.len;
    const heat_ray = HeatRay{ .damage = 7 }; // We've been given a heat ray weapon.

    // We'll keep checking to see if we've killed all the aliens yet.
    while (aliens_alive > 0) {
        aliens_alive = 0;

        // Loop through every alien by reference (* makes a pointer capture value)
        for (&aliens) |*alien| {

            // *** Zap the alien with the heat ray here! ***
            heat_ray.zap(alien);

            // If the alien's health is still above 0, it's still alive.
            if (alien.health > 0) aliens_alive += 1;
        }

        std.debug.print("{} aliens. ", .{aliens_alive});
    }

    std.debug.print("Earth is saved!\n", .{});
}
```

## 048_methods2.zig

```zig
fn visitElephants(first_elephant: *Elephant) void {
    var e = first_elephant;

    while (true) {
        e.print();
        e.visit();

        // This gets the next elephant or stops:
        // which method do we want here?
        e = if (e.hasTail()) e.getTail() else break;
    }
}
```

## 049_quiz6.zig

```zig
    // Your Elephant trunk methods go here!
    // ---------------------------------------------------
    pub fn getTrunk(self: *Elephant) *Elephant {
        return self.trunk.?; // Remember, this means "orelse unreachable"
    }

    pub fn hasTrunk(self: *Elephant) bool {
        return (self.trunk != null);
    }
```

## 050_no_value.zig

```zig
const Err = error{Cthulhu};

pub fn main() void {
    var first_line1: *const [16]u8 = undefined;
    first_line1 = "That is not dead";

    var first_line2: Err!*const [21]u8 = undefined;
    first_line2 = "which can eternal lie";

    // Note we need the "{!s}" format for the error union string.
    std.debug.print("{s} {!s} / ", .{ first_line1, first_line2 });

    printSecondLine();
}

fn printSecondLine() void {
    var second_line2: ?*const [18]u8 = undefined;
    second_line2 = "even death may die";

    std.debug.print("And with strange aeons {s}.\n", .{second_line2.?});
}
```

## 051_values.zig

```zig
pub fn main() void {

    ..

    var glorp = Character{
        .gold = 30,
    };

    print("XP before:{}, ", .{glorp.experience});

    // Fix 1 of 2 goes here:
    levelUp(&glorp, reward_xp);

    print("after:{}.\n", .{glorp.experience});
}

// Fix 2 of 2 goes here:
fn levelUp(character_access: *Character, xp: u32) void {
    character_access.experience += xp;
}
```

## 052_slices.zig

```zig
pub fn main() void {
    var cards = [8]u8{ 'A', '4', 'K', '8', '5', '2', 'Q', 'J' };

    // Please put the first 4 cards in hand1 and the rest in hand2.
    const hand1: []u8 = cards[0..4];
    const hand2: []u8 = cards[4..8];

    std.debug.print("Hand1: ", .{});
    printHand(hand1);

    std.debug.print("Hand2: ", .{});
    printHand(hand2);
}
```

## 053_slices2.zig

```zig
pub fn main() void {
    const scrambled = "great base for all your justice are belong to us";

    const base1: []const u8 = scrambled[15..23];
    const base2: []const u8 = scrambled[6..10];
    const base3: []const u8 = scrambled[32..];
    printPhrase(base1, base2, base3);

    const justice1: []const u8 = scrambled[11..14];
    const justice2: []const u8 = scrambled[0..5];
    const justice3: []const u8 = scrambled[24..31];
    printPhrase(justice1, justice2, justice3);

    std.debug.print("\n", .{});
}

fn printPhrase(part1: []const u8, part2: []const u8, part3: []const u8) void {
    std.debug.print("'{s} {s} {s}.' ", .{ part1, part2, part3 });
}
```

## 054_manypointers.zig

```zig
    const zen12: *const [21]u8 = "Memory is a resource.";
    const zen_manyptr: [*]const u8 = zen12;
    const zen12_string: []const u8 = zen_manyptr[0..21];

    // Here's the moment of truth!
    std.debug.print("{s}\n", .{zen12_string});
```

## 055_unions.zig

```zig
const Insect = union {
    flowers_visited: u16,
    still_alive: bool,
};
const AntOrBee = enum { a, b };

pub fn main() void {
    // We'll just make one bee and one ant to test them out:
    const ant = Insect{ .still_alive = true };
    const bee = Insect{ .flowers_visited = 15 };

    std.debug.print("Insect report! ", .{});

    // Oops! We've made a mistake here.
    printInsect(ant, AntOrBee.a);
    printInsect(bee, AntOrBee.b);

    std.debug.print("\n", .{});
}

fn printInsect(insect: Insect, what_it_is: AntOrBee) void {
    switch (what_it_is) {
        .a => std.debug.print("Ant alive is: {}. ", .{insect.still_alive}),
        .b => std.debug.print("Bee visited {} flowers. ", .{insect.flowers_visited}),
    }
}
```

## 056_unions2.zig

```zig
const InsectStat = enum { flowers_visited, still_alive };

const Insect = union(InsectStat) {
    flowers_visited: u16,
    still_alive: bool,
};

pub fn main() void {
    const ant = Insect{ .still_alive = true };
    const bee = Insect{ .flowers_visited = 16 };

    std.debug.print("Insect report! ", .{});

    // Could it really be as simple as just passing the union?
    printInsect(ant);
    printInsect(bee);

    std.debug.print("\n", .{});
}

fn printInsect(insect: Insect) void {
    switch (insect) {
        .still_alive => |a| std.debug.print("Ant alive is: {}. ", .{a}),
        .flowers_visited => |f| std.debug.print("Bee visited {} flowers. ", .{f}),
    }
}
```

## 057_unions3.zig

```zig
const Insect = union(enum) {
    flowers_visited: u16,
    still_alive: bool,
};

fn printInsect(insect: Insect) void {
    switch (insect) {
        .still_alive => |a| std.debug.print("Ant alive is: {}. ", .{a}),
        .flowers_visited => |f| std.debug.print("Bee visited {} flowers. ", .{f}),
    }
}
```

## 058_quiz7.zig

```zig
    // found, we return null.
    fn getEntry(self: *HermitsNotebook, place: *const Place) ?*NotebookEntry {
        for (&self.entries, 0..) |*entry, i| {
            if (i >= self.end_of_entries) break;
            if (place == entry.*.?.place) return &entry.*.?;
        }
        return null;
    }
```

## 059_integers.zig

```zig
pub fn main() void {
    const zig = [_]u8{
        0o132, // octal
        0b01101001, // binary
        0x67, // hex
    };

    print("{s} is cool.\n", .{zig});
}
```

## 060_floats.zig

```zig
    const shuttle_weight: f32 = 0.453592 * 4480000;

    // By default, float values are formatted in scientific
    // notation. Try experimenting with '{d}' and '{d:.3}' to see
    // how decimal formatting works.
    print("Shuttle liftoff weight: {d:.0}kg\n", .{shuttle_weight});
```

## 061_coercions.zig

```zig
pub fn main() void {
    var letter: u8 = 'A';

    const my_letter: ?*[1]u8 = &letter;
    //               ^^^^^^^
    //           Your type here.
    // Must coerce from &letter (which is a *u8).
    // Hint: Use coercion Rules 4 and 5.

    // When it's right, this will work:
    print("Letter: {u}\n", .{my_letter.?.*[0]});
}
```

## 062_loop_expressions.zig

```zig
pub fn main() void {
    const langs: [6][]const u8 = .{
        "Erlang",
        "Algol",
        "C",
        "OCaml",
        "Zig",
        "Prolog",
    };

    // Let's find the first language with a three-letter name and
    // return it from the for loop.
    const current_lang: ?[]const u8 = for (langs) |lang| {
        if (lang.len == 3) break lang;
    } else "";

    // current_lang is a ?[]const u8, so we need to check it (value or null).
    // if it's not null, cl gets the value and we print it.
    // if it's null, we print a message that didn't find a match.
    if (current_lang) |cl| {
        print("Current language: {s}\n", .{cl});
    } else {
        print("Did not find a three-letter language name. :-(\n", .{});
    }

    // Here is the same thing, but using the if-null operator.
    // this not explicit about the type of current_lang.
    // I have to check it explicitly inside if.
    if (current_lang != null) {
        print("Current language: {s}\n", .{current_lang.?});
    } else {
        print("Did not find a three-letter language name. :-(\n", .{});
    }
}
```

## 063_labels.zig

```zig
const ingredients = 4;
const foods = 4;

const Food = struct {
    name: []const u8,
    requires: [ingredients]bool,
};

const menu: [foods]Food = [_]Food{
    Food{
        .name = "Mac & Cheese",
        .requires = [ingredients]bool{ false, true, false, true },
    },
};

pub fn main() void {
    const wanted_ingredients = [_]u8{ 0, 3 }; // Chili, Cheese

    // Look at each Food on the menu...
    const meal: Food = food_loop: for (menu) |food| {

        // Now look at each required ingredient for the Food...
        for (food.requires, 0..) |required, required_ingredient| {

            // This ingredient isn't required, so skip it.
            if (!required) continue;

            // See if the customer wanted this ingredient.
            // (Remember that want_it will be the index number of
            // the ingredient based on its position in the
            // required ingredient list for each food.)
            const found = for (wanted_ingredients) |want_it| {
                if (required_ingredient == want_it) break true;
            } else false;

            // We did not find this required ingredient, so we
            // can't make this Food. Continue the outer loop.
            if (!found) continue :food_loop;
        }

        break food;
    } else menu[0];

    print("Enjoy your {s}!\n", .{meal.name});
}
```

## 064_builtins.zig

```zig
    const expected_result: u8 = 0b10010;
    print(". Without overflow: {b:0>8}. ", .{expected_result});

    print("Furthermore, ", .{});

    // Here's a fun one:
    //
    //   @bitReverse(integer: anytype) T
    //     * 'integer' is the value to reverse.
    //     * The return value will be the same type with the
    //       value's bits reversed!
    //
    // Now it's your turn. See if you can fix this attempt to use
    // this builtin to reverse the bits of a u8 integer.
    const input: u8 = 0b11110000;
    const tupni: u8 = @bitReverse(input);
    print("{b:0>8} backwards is {b:0>8}.\n", .{ input, tupni });
```

## 065_builtins2.zig

```zig
const Narcissus = struct {
    me: *Narcissus = undefined,
    myself: *Narcissus = undefined,
    echo: void = undefined, // Alas, poor Echo!

   plain fn fetchTheMostBeautifulType() type {
        return @This();
    }
};

pub fn main() void {

    var narcissus: Narcissus = Narcissus{};

    narcissus.me = &narcissus;
    narcissus.myself = &narcissus;

    // Oh dear, we seem to have done something wrong when calling
    // this function. We called it as a method, which would work
    // if it had a self parameter. But it doesn't. (See above.)
    //
    // The fix for this is very subtle, but it makes a big
    // difference!
    const Type2 = Narcissus.fetchTheMostBeautifulType();

    // 'fields' is a slice of StructFields. Here's the declaration:
    //
    //     pub const StructField = struct {
    //         name: []const u8,
    //         type: type,
    //         default_value: anytype,
    //         is_comptime: bool,
    //         alignment: comptime_int,
    //     };
    //
    // Please complete these 'if' statements so that the field
    // name will not be printed if the field is of type 'void'
    // (which is a zero-bit type that takes up no space at all!):
    if (fields[0].type != void) {
        print(" {s}", .{@typeInfo(Narcissus).Struct.fields[0].name});
    }
}
```

## 066_comptime.zig

```zig
    const const_int = 12345;
    const const_float = 987.654;

    var var_int: u32 = 12345;
    var var_float: f32 = 987.654;
```

## 067_comptime2.zig

```zig
   comptime var count = 0;

    // Builtin BONUS!
    //
    // The @compileLog() builtin is like a print statement that
    // ONLY operates at compile time. The Zig compiler treats
    // @compileLog() calls as errors, so you'll want to use them
    // temporarily to debug compile time logic.
    //
    // Try uncommenting this line and playing around with it
    // (copy it, move it) to see what it does:
    @compileLog("Count at compile time: ", count);
```

## 068_comptime3.zig

```zig
pub fn main() void {
    var whale = Schooner{ .name = "Whale" };
    var shark = Schooner{ .name = "Shark" };
    var minnow = Schooner{ .name = "Minnow" };

    // Hey, we can't just pass this runtime variable as an
    // argument to the scaleMe() method. What would let us do
    // that?
    comptime var scale: u32 = undefined;

    scale = 32; // 1:32 scale

    minnow.scaleMe(scale);
    minnow.printMe();

    scale -= 16; // 1:16 scale

    shark.scaleMe(scale);
    shark.printMe();

    scale -= 16; // 1:0 scale (oops, but DON'T FIX THIS!)

    whale.scaleMe(scale);
    whale.printMe();
}
```

## 069_comptime4.zig

```zig
pub fn main() void {
    // Here we declare arrays of three different types and sizes
    // at compile time from a function call. Neat!
    const s1 = makeSequence(u8, 3); // creates a [3]u8
    const s2 = makeSequence(u32, 5); // creates a [5]u32
    const s3 = makeSequence(i64, 7); // creates a [7]i64

    print("s1={any}, s2={any}, s3={any}\n", .{ s1, s2, s3 });
}

fn makeSequence(comptime T: type, comptime size: usize) [size]T {
    var sequence: [size]T = undefined;
    var i: usize = 0;

    while (i < size) : (i += 1) {
        sequence[i] = @as(T, @intCast(i)) + 1;
    }

    return sequence;
}
```

## 070_comptime5.zig

```zig
fn isADuck(possible_duck: anytype) bool {
    // We'll use @hasDecl() to determine if the type has
    // everything needed to be a "duck".
    //
    // In this example, 'has_increment' will be true if type Foo
    // has an increment() method:
    //
    //     const has_increment = @hasDecl(Foo, "increment");
    //
    // Please make sure MyType has both waddle() and quack()
    // methods:
    const MyType = @TypeOf(possible_duck);
    const walks_like_duck = @hasDecl(MyType, "waddle");
    const quacks_like_duck = @hasDecl(MyType, "quack");

    const is_duck = walks_like_duck and quacks_like_duck;

    if (is_duck) {
        // We also call the quack() method here to prove that Zig
        // allows us to perform duck actions on anything
        // sufficiently duck-like.
        //
        // Because all of the checking and inference is performed
        // at compile time, we still have complete type safety:
        // attempting to call the quack() method on a struct that
        // doesn't have it (like Duct) would result in a compile
        // error, not a runtime panic or crash!
        possible_duck.quack();
    }

    return is_duck;
}
```

## 071_comptime6.zig

```zig
pub fn main() void {
    print("Narcissus has room in his heart for:", .{});

    // Last time we examined the Narcissus struct, we had to
    // manually access each of the three fields. Our 'if'
    // statement was repeated three times almost verbatim. Yuck!
    //
    // Please use an 'inline for' to implement the block below
    // for each field in the slice 'fields'!

    const fields = @typeInfo(Narcissus).Struct.fields;

    inline for (fields) |field| {
        if (field.type != void) {
            print(" {s}", .{field.name});
        }
    }
    // Once you've got that, go back and take a look at exercise
    // 065 and compare what you've written to the abomination we
    // had there!

    print(".\n", .{});
}
```

## 072_comptime7.zig

```zig
pub fn main() void {
    // Here is a string containing a series of arithmetic
    // operations and single-digit decimal values. Let's call
    // each operation and digit pair an "instruction".
    const instructions = "+3 *5 -2 *2";

    // Here is a u32 variable that will keep track of our current
    // value in the program at runtime. It starts at 0, and we
    // will get the final value by performing the sequence of
    // instructions above.
    var value: u32 = 0;

    // This "index" variable will only be used at compile time in
    // our loop.
    comptime var i = 0;

    // Here we wish to loop over each "instruction" in the string
    // at compile time.
    inline while (i < instructions.len) : (i += 3) {

        // This gets the digit from the "instruction". Can you
        // figure out why we subtract '0' from it?
        const digit = instructions[i + 1] - '0';

        // This 'switch' statement contains the actual work done
        // at runtime. At first, this doesn't seem exciting...
        switch (instructions[i]) {
            '+' => value += digit,
            '-' => value -= digit,
            '*' => value *= digit,
            else => unreachable,
        }
    }

    print("{}\n", .{value});
}
```

## 073_comptime8.zig

```zig
const llama_count = 5;
const llamas = [llama_count]u32{ 5, 10, 15, 20, 25 };

pub fn main() void {
    // We meant to fetch the last llama. Please fix this simple
    // mistake so the assertion no longer fails.
    const my_llama = getLlama(4);

    print("My llama value is {}.\n", .{my_llama});
}

fn getLlama(comptime i: usize) u32 {
    comptime assert(i < llama_count);
    return llamas[i];
}

// Fun fact: this assert() function is identical to
// std.debug.assert() from the Zig Standard Library.
fn assert(ok: bool) void {
    if (!ok) unreachable;
}
```

## 074_comptime9.zig

```zig
// Being in the container-level scope, everything about this value is
// implicitly required to be known compile time.
const llama_count = 5;

// Again, this value's type and size must be known at compile
// time, but we're letting the compiler infer both from the
// return type of a function.
const llamas = makeLlamas(llama_count);

// And here's the function. Note that the return value type
// depends on one of the input arguments!
fn makeLlamas(comptime count: usize) [count]u8 {
    var temp: [count]u8 = undefined;
    var i = 0;

    // Note that this does NOT need to be an inline 'while'.
    while (i < count) : (i += 1) {
        temp[i] = i;
    }

    return temp;
}

pub fn main() void {
    print("My llama value is {}.\n", .{llamas[2]});
}
```

## 075_quiz8.zig

```zig
fn makePath(from: *Place, to: *Place, dist: u8) Path {
    return Path{
        .from = from,
        .to = to,
        .dist = dist,
    };
}

// Using our new function, these path definitions take up considerably less
// space in our program now!
const a_paths = [_]Path{makePath(&a, &b, 2)};
const b_paths = [_]Path{ makePath(&b, &a, 2), makePath(&b, &d, 1) };
const c_paths = [_]Path{ makePath(&c, &d, 3), makePath(&c, &e, 2) };
```

## 076_sentinels.zig

```zig
fn printSequence(my_seq: anytype) void {
    const my_typeinfo = @typeInfo(@TypeOf(my_seq));

    // The TypeInfo contained in my_typeinfo is a union. We use
    // a switch to handle printing the Array or Pointer fields,
    // depending on which type of my_seq was passed in:
    switch (my_typeinfo) {
        .Array => {
            print("Array:", .{});

            // Loop through the items in my_seq.
            for (???) |s| {
                print("{}", .{s});
            }
        },
        .Pointer => {
            // Check this out - it's pretty cool:
            const my_sentinel = sentinel(@TypeOf(my_seq));
            print("Many-item pointer:", .{});

            // Loop through the items in my_seq until we hit the
            // sentinel value.
            var i: usize = 0;
            while (??? != my_sentinel) {
                print("{}", .{my_seq[i]});
                i += 1;
            }
        },
        else => unreachable,
    }
    print(". ", .{});
}
```
## 077_sentinels2.zig

```zig
pub fn main() void {
    const foo = WeirdContainer{
        .data = "Weird Data!",
        .length = 11,
    };

    // Here's a big hint: do you remember how to take a slice?
    const printable = foo.data[0..foo.length];

    print("{s}\n", .{printable});
```

## 078_sentinels3.zig

```zig
pub fn main() void {
    // Again, we've coerced the sentinel-terminated string to a
    // many-item pointer, which has no length or sentinel.
    const data: [*]const u8 = "Weird Data!";

    // Please cast 'data' to 'printable':
    const printable: [*:0]const u8 = @ptrCast(data);

    print("{s}\n", .{printable});
}
```

## 079_quoted_identifiers.zig

```zig
pub fn main() void {
    const @"55_cows": i32 = 55;
    const @"isn't true": bool = false;

    print("Sweet freedom: {}, {}.\n", .{
        @"55_cows",
        @"isn't true",
    });
}
```

## 080_anonymous_structs.zig

```zig
fn Circle(comptime T: type) type {
    return struct {
        center_x: T,
        center_y: T,
        radius: T,
    };
}

pub fn main() void {
    //
    // See if you can complete these two variable initialization
    // expressions to create instances of circle struct types
    // which can hold these values:
    //
    // * circle1 should hold i32 integers
    // * circle2 should hold f32 floats
    //
    const circle1 = Circle(i32){
        .center_x = 25,
        .center_y = 70,
        .radius = 15,
    };

    const circle2 = Circle(f32){
        .center_x = 25.234,
        .center_y = 70.999,
        .radius = 15.714,
    };
}
```

## 081_anonymous_structs2.zig

```zig
pub fn main() void {
    printCircle(.{
        .center_x = @as(u32, 205),
        .center_y = @as(u32, 187),
        .radius = @as(u32, 12),
    });
}

// Please complete this function which prints an anonymous struct
// representing a circle.
fn printCircle(circle: anytype) void {
    print("x:{} y:{} radius:{}\n", .{
        circle.center_x,
        circle.center_y,
        circle.radius,
    });
}
```

## 082_anonymous_structs3.zig

```zig
pub fn main() void {
    // A "tuple":
    const foo = .{
        true,
        false,
        @as(i32, 42),
        @as(f32, 3.141592),
    };

    // We'll be implementing this:
    printTuple(foo);

    // This is just for fun, because we can:
    const nothing = .{};
    print("\n", nothing);
}

fn printTuple(tuple: anytype) void {
    // This will be an array of StructFields.
    const fields = @typeInfo(@TypeOf(tuple)).Struct.fields;

    // 2. Loop through each field. This must be done at compile
    // time.
    inline for (fields) |field| {
        // The first field should print as: "0"(bool):true
        print("\"{s}\"({any}):{any} ", .{
            field.name,
            field.type,
            @field(tuple, field.name),
        });
    }
}

```

## 083_anonymous_lists.zig

```zig
pub fn main() void {
    // Please make 'hello' a string-like array of u8 WITHOUT
    // changing the value literal.
    //
    // Don't change this part:
    //
    //     = .{ 'h', 'e', 'l', 'l', 'o' };
    //
    const hello: [5]u8 = .{ 'h', 'e', 'l', 'l', 'o' };
    print("I say {s}!\n", .{hello});
}
```
## 084_async.zig to 091_async8.zig

async was removed in zig 0.10.0

## 092_interfaces.zig

```zig
const Insect = union(enum) {
    ant: Ant,
    bee: Bee,
    grasshopper: Grasshopper,

    // Thanks to 'inline else', we can think of this print() as
    // being an interface method. Any member of this union with
    // a print() method can be treated uniformly by outside
    // code without needing to know any other details. Cool!
    pub fn print(self: Insect) void {
        switch (self) {
            inline else => |case| return case.print(),
        }
    }
};

pub fn main() !void {
    const my_insects = [_]Insect{
        Insect{ .ant = Ant{ .still_alive = true } },
        Insect{ .bee = Bee{ .flowers_visited = 17 } },
        Insect{ .grasshopper = Grasshopper{ .distance_hopped = 32 } },
    };

    std.debug.print("Daily Insect Report:\n", .{});
    for (my_insects) |insect| {
        // Almost done! We want to print() each insect with a
        // single method call here.
        insect.print();
    }
}
```

## 093_hello_c.zig

```zig
// and here the new import for C
const c = @cImport({
    @cInclude("unistd.h");
});

pub fn main() void {

    // In order to output text that can be evaluated by the
    // Zig Builder, we need to write it to the Error output.
    // In Zig, we do this with "std.debug.print" and in C we can
    // specify a file descriptor i.e. 2 for error console.
    //
    // In this exercise we use 'write' to output 17 chars,
    // but something is still missing...
    const c_res = c.write(2, "Hello C from Zig!", 17);

    // let's see what the result from C is:
    std.debug.print(" - C result is {d} chars written.\n", .{c_res});
}
```

## 094_c_math.zig

```zig
const c = @cImport({
    // What do we need here?
    @cInclude("math.h");
});

pub fn main() !void {
    const angle = 765.2;
    const circle = 360;

    // Here we call the C function 'fmod' to get our normalized angle.
    const result = c.fmod(angle, circle);

    // We use formatters for the desired precision and to truncate the decimal places
    std.debug.print("The normalized angle of {d: >3.1} degrees is {d: >3.1} degrees.\n", .{ angle, result });
}
```

## 095_for3.zig

```zig
pub fn main() void {

    // I want to print every number between 1 and 20 that is NOT
    // divisible by 3 or 5.
    for (0..21) |n| {

        // The '%' symbol is the "modulo" operator and it
        // returns the remainder after division.
        if (n % 3 == 0) continue;
        if (n % 5 == 0) continue;
        std.debug.print("{} ", .{n});
    }

    std.debug.print("\n", .{});
}
```

## 096_memory_allocation.zig

```zig
pub fn main() !void {
    // pretend this was defined by reading in user input
    const arr: []const f64 = &[_]f64{ 0.3, 0.2, 0.1, 0.1, 0.4 };

    // initialize the allocator
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);

    // free the memory on exit
    defer arena.deinit();

    // initialize the allocator
    const allocator = arena.allocator();

    // allocate memory for this array
    const avg: []f64 = try allocator.alloc(f64, arr.len);

    runningAverage(arr, avg);
    std.debug.print("Running Average: ", .{});
    for (avg) |val| {
        std.debug.print("{d:.2} ", .{val});
    }
    std.debug.print("\n", .{});
}
```

## 097_bit_manipulation.zig

```zig
pub fn main() !void {

    // As in the example above, we use 1 and 0 as values for x and y
    var x: u8 = 1;
    var y: u8 = 0;

    // Now we swap the values of the two variables by doing xor on them
    x ^= y;
    y ^= x;

    // What must be written here?
    x ^= y;

    print("x = {d}; y = {d}\n", .{ x, y });
}
```

## 098_bit_manipulation2.zig

```zig
fn isPangram(str: []const u8) bool {
    // first we check if the string has at least 26 characters
    if (str.len < 26) return false;

    // we use a 32 bit variable of which we need 26 bits
    var bits: u32 = 0;

    // loop about all characters in the string
    for (str) |c| {
        // if the character is an alphabetical character
        if (ascii.isASCII(c) and ascii.isAlphabetic(c)) {
            // then we set the bit at the position
            //
            // to do this, we use a little trick:
            // since the letters in the ASCII table start at 65
            // and are numbered sequentially, we simply subtract the
            // first letter (in this case the 'a') from the character
            // found, and thus get the position of the desired bit
            bits |= @as(u32, 1) << @truncate(ascii.toLower(c) - 'a');
        }
    }
    // last we return the comparison if all 26 bits are set,
    // and if so, we know the given string is a pangram
    //
    // but what do we have to compare?
    return bits == 0x3FFFFFF;
}
```

## 099_formatting.zig

```zig
pub fn main() !void {
    // Max number to multiply
    const size = 15;

    // Print the header:
    //
    // We start with a single 'X' for the diagonal.
    print("\n X |", .{});

    // Header row with all numbers from 1 to size.
    for (0..size) |n| {
        print("{d:>3} ", .{n + 1});
    }
    print("\n", .{});

    // Header column rule line.
    var n: u8 = 0;
    while (n <= size) : (n += 1) {
        print("---+", .{});
    }
    print("\n", .{});

    // Now the actual table. (Is there anything more beautiful
    // than a well-formatted table?)
    for (0..size) |a| {
        print("{d:>2} |", .{a + 1});

        for (0..size) |b| {
            // What formatting is needed here to make our columns
            // nice and straight?
            print("{d:>3} ", .{(a + 1) * (b + 1)});
        }

        // After each row we use double line feed:
        print("\n\n", .{});
    }
}
```

## 100_for4.zig

```zig
pub fn main() void {
    const hex_nums = [_]u8{ 0xb, 0x2a, 0x77 };
    const dec_nums = [_]u8{ 11, 42, 119 };

    for (hex_nums, dec_nums) |hn, dn| {
        if (hn != dn) {
            std.debug.print("Uh oh! Found a mismatch: {d} vs {d}\n", .{ hn, dn });
            return;
        }
    }

    std.debug.print("Arrays match!\n", .{});
}
```

## 101_for5.zig

```zig
pub fn main() void {
    // Here are the three "property" arrays:
    const roles = [4]Role{ .wizard, .bard, .bard, .warrior };
    const gold = [4]u16{ 25, 11, 5, 7392 };
    const experience = [4]u8{ 40, 17, 55, 21 };

    // We would like to number our list starting with 1, not 0.
    // How do we do that?
    for (roles, gold, experience, 1..) |c, g, e, i| {
        const role_name = switch (c) {
            .wizard => "Wizard",
            .thief => "Thief",
            .bard => "Bard",
            .warrior => "Warrior",
        };

        std.debug.print("{d}. {s} (Gold: {d}, XP: {d})\n", .{
            i,
            role_name,
            g,
            e,
        });
    }
}
```

## 102_testing.zig

```zig
fn sub(a: f16, b: f16) f16 {
    return a - b;
}

test "sub" {
    try testing.expect(sub(10, 5) == 5);

    try testing.expect(sub(3, 1.5) == 1.5);
}
fn divide(a: f16, b: f16) !f16 {
    if (b == 0) return error.DivisionByZero;
    return a / b;
}

test "divide" {
    try testing.expect(divide(2, 2) catch unreachable == 1);
    try testing.expect(divide(-1, -1) catch unreachable == 1);
    try testing.expect(divide(10, 2) catch unreachable == 5);
    try testing.expect(divide(1, 3) catch unreachable == 0.3333333333333333);

    // Now we test if the function returns an error
    // if we pass a zero as denominator.
    // But which error needs to be tested?
    try testing.expectError(error.DivisionByZero, divide(15, 0));
}
```
## 103_tokenization.zig

```zig
pub fn main() !void {

    // our input
    const poem =
        \\My name is Ozymandias, King of Kings;
        \\Look on my Works, ye Mighty, and despair!
    ;

    // now the tokenizer, but what do we need here?
    var it = std.mem.tokenizeAny(u8, poem, " ,;!\n");

    // print all words and count them
    var cnt: usize = 0;
    while (it.next()) |word| {
        cnt += 1;
        print("{s}\n", .{word});
    }

    // print the result
    print("This little poem has {d} words!\n", .{cnt});
}
```

## 104_threading.zig

```zig
pub fn main() !void {
    // This is where the preparatory work takes place
    // before the parallel processing begins.
    std.debug.print("Starting work...\n", .{});

    // These curly brackets are very important, they are necessary
    // to enclose the area where the threads are called.
    // Without these brackets, the program would not wait for the
    // end of the threads and they would continue to run beyond the
    // end of the program.
    {
        // Now we start the first thread, with the number as parameter
        const handle = try std.Thread.spawn(.{}, thread_function, .{1});

        // Waits for the thread to complete,
        // then deallocates any resources created on `spawn()`.
        defer handle.join();

        // Second thread
        const handle2 = try std.Thread.spawn(.{}, thread_function, .{2}); // that can't be right?
        defer handle2.join();

        // Third thread
        const handle3 = try std.Thread.spawn(.{}, thread_function, .{3});
        defer handle3.join(); // <-- something is missing

        // After the threads have been started,
        // they run in parallel and we can still do some work in between.
        std.time.sleep(1500 * std.time.ns_per_ms);
        std.debug.print("Some weird stuff, after starting the threads.\n", .{});
    }
    // After we have left the closed area, we wait until
    // the threads have run through, if this has not yet been the case.
    std.debug.print("Zig is cool!\n", .{});
}

// This function is started with every thread that we set up.
// In our example, we pass the number of the thread as a parameter.
fn thread_function(num: usize) !void {
    std.time.sleep(200 * num * std.time.ns_per_ms);
    std.debug.print("thread {d}: {s}\n", .{ num, "started." });

    // This timer simulates the work of the thread.
    const work_time = 3 * ((5 - num % 3) - 2);
    std.time.sleep(work_time * std.time.ns_per_s);

    std.debug.print("thread {d}: {s}\n", .{ num, "finished." });
}
```

## 105_threading2.zig

```zig
pub fn main() !void {
    const count = 1_000_000_000;
    var pi_plus: f64 = 0;
    var pi_minus: f64 = 0;

    {
        // First thread to calculate the plus numbers.
        const handle1 = try std.Thread.spawn(.{}, thread_pi, .{ &pi_plus, 5, count });
        defer handle1.join();

        // Second thread to calculate the minus numbers.
        const handle2 = try std.Thread.spawn(.{}, thread_pi, .{ &pi_minus, 3, count });
        defer handle2.join();
    }
    // Here we add up the results.
    std.debug.print("PI ≈ {d:.8}\n", .{4 + pi_plus - pi_minus});
}

fn thread_pi(pi: *f64, begin: u64, end: u64) !void {
    var n: u64 = begin;
    while (n < end) : (n += 4) {
        pi.* += 4 / @as(f64, @floatFromInt(n));
    }
}
```

## 106_files.zig

```zig
pub fn main() !void {
    // first we get the current working directory
    const cwd: std.fs.Dir = std.fs.cwd();

    // then we'll try to make a new directory /output/
    // to store our output files.
    cwd.makeDir("output") catch |e| switch (e) {

        error.PathAlreadyExists => {},
        // if there's any other unexpected error we just propagate it through
        else => return e,
    };

    // then we'll try to open our freshly created directory
    // wait a minute...
    // opening a directory might fail!
    // what should we do here?
    var output_dir: std.fs.Dir = cwd.openDir("output", .{});
    defer output_dir.close();

    // we try to open the file `zigling.txt`,
    // and propagate any error up
    var output_dir: std.fs.Dir = cwd.openDir("output", .{}) catch |e| {
        return e;
    };
    file.close();

    // you are not allowed to move these two lines above the file closing line!
    const byte_written = try file.write("It's zigling time!");
    std.debug.print("Successfully wrote {d} bytes.\n", .{byte_written});
}
```

## 107_files2.zig

```zig
pub fn main() !void {
    // Get the current working directory
    const cwd = std.fs.cwd();

    // try to open ./output assuming you did your 106_files exercise
    var output_dir = try cwd.openDir("output", .{});
    defer output_dir.close();

    // try to open the file
    const file = try output_dir.openFile("zigling.txt", .{});
    defer file.close();

    // initalize an array of u8 with all letter 'A'
    // we need to pick the size of the array, 64 seems like a good number
    // fix the initalization below
    var content = [_]u8{'A'} ** 64;
    // this should print out : `AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA`
    std.debug.print("{s}\n", .{content});
    const bytes_read = try file.readAll(&content);
    std.debug.print("Successfully Read {d} bytes: {s}\n", .{
        bytes_read,
        content[0..bytes_read], // change this line only
    });
}
```
