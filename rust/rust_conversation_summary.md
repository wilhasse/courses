# Rust Language Context With Examples

This file keeps the principal language ideas from the transcript and adds small examples in an order that builds from basic type-safety concepts to Rust's ownership model.

## 1. Reliability as the Main Theme

The transcript presents Rust as a language that tries to catch common mistakes before the program runs. The phrase "when it compiles, it works" is not literal, but it describes the feeling that the compiler checks many things other languages leave for runtime.

Rust does this through:

- strong static types;
- explicit missing values;
- explicit errors;
- exhaustive pattern matching;
- ownership;
- borrowing;
- memory safety;
- controlled `unsafe`.

Simple example:

```rust
fn double(number: i32) -> i32 {
    number * 2
}

fn main() {
    let value = double(10);
    println!("{value}");
}
```

Here the function accepts only an `i32`. If another type is passed, the compiler rejects the program before it runs.

## 2. Type System and Null Safety

The transcript contrasts Rust with languages where a value may unexpectedly be `null`. Rust does not use implicit null values. If something may be missing, the type says so with `Option<T>`.

```rust
fn find_user_name(id: u32) -> Option<String> {
    if id == 1 {
        Some(String::from("Alice"))
    } else {
        None
    }
}

fn main() {
    let name = find_user_name(2);

    match name {
        Some(value) => println!("User: {value}"),
        None => println!("User not found"),
    }
}
```

The important point is that the caller must handle both cases. There is no hidden null crash.

## 3. Explicit Error Handling

Rust commonly represents fallible operations with `Result<T, E>`. This makes failure part of the function signature.

```rust
fn divide(a: i32, b: i32) -> Result<i32, String> {
    if b == 0 {
        Err(String::from("cannot divide by zero"))
    } else {
        Ok(a / b)
    }
}

fn main() {
    match divide(10, 0) {
        Ok(value) => println!("Result: {value}"),
        Err(error) => println!("Error: {error}"),
    }
}
```

The transcript emphasizes that Rust avoids hidden exception paths. The possibility of failure is visible in the type.

## 4. The `?` Operator

The `?` operator makes explicit error handling less repetitive. If the result is `Err`, the error is returned from the current function. If it is `Ok`, the value is unwrapped.

```rust
fn parse_number(text: &str) -> Result<i32, std::num::ParseIntError> {
    let number = text.parse::<i32>()?;
    Ok(number)
}

fn main() {
    match parse_number("42") {
        Ok(number) => println!("Parsed: {number}"),
        Err(error) => println!("Invalid number: {error}"),
    }
}
```

The transcript presents this as a balance: error handling is still explicit, but the code stays compact.

## 5. Exhaustive `match`

Rust's `match` is compared with `switch`, but with a stronger rule: every possible case must be handled.

```rust
enum PaymentStatus {
    Paid,
    Pending,
    Failed,
}

fn message(status: PaymentStatus) -> &'static str {
    match status {
        PaymentStatus::Paid => "payment confirmed",
        PaymentStatus::Pending => "payment pending",
        PaymentStatus::Failed => "payment failed",
    }
}

fn main() {
    println!("{}", message(PaymentStatus::Paid));
}
```

If a new variant is added to `PaymentStatus`, the compiler points to every `match` that must be updated. This is one way Rust helps with refactoring.

## 6. Documentation as Tests

The transcript highlights Rust documentation examples because they can be compiled and run as tests. This helps prevent examples from becoming outdated.

```rust
/// Adds two numbers.
///
/// ```
/// let result = add(2, 3);
/// assert_eq!(result, 5);
/// ```
fn add(a: i32, b: i32) -> i32 {
    a + b
}
```

When `cargo test` runs, Rust can test the example in the documentation. If the function changes and the example breaks, the test fails.

## 7. No Garbage Collector

Rust does not use a garbage collector. Values are normally cleaned up when their owner goes out of scope.

```rust
fn main() {
    {
        let name = String::from("Alice");
        println!("{name}");
    } // `name` is cleaned up here

    println!("scope ended");
}
```

The transcript explains that this gives predictable resource management. Rust can be used in low-level or latency-sensitive contexts because cleanup is tied to scope instead of a runtime garbage-collection pass.

## 8. Ownership

Ownership means each value has one owner at a time. Moving a value transfers ownership.

```rust
fn main() {
    let a = String::from("hello");
    let b = a;

    println!("{b}");
    // println!("{a}"); // error: `a` was moved
}
```

The transcript explains this through cleanup. If both `a` and `b` owned the same string, Rust would not know which one should free it. After the move, only `b` owns the value.

## 9. Borrowing and References

Borrowing lets code use a value without taking ownership.

```rust
fn print_length(text: &String) {
    println!("length: {}", text.len());
}

fn main() {
    let name = String::from("Alice");

    print_length(&name);
    println!("still usable: {name}");
}
```

`print_length` receives a reference. It can read the string, but it does not own it. After the function call, `name` is still valid.

## 10. Mutable and Immutable Borrows

The borrow checker enforces one of Rust's central rules: many readers or one writer, but not both at the same time.

```rust
fn main() {
    let mut name = String::from("Alice");

    let read1 = &name;
    let read2 = &name;
    println!("{read1} and {read2}");

    let write = &mut name;
    write.push_str(" Smith");
    println!("{write}");
}
```

The immutable borrows are used before the mutable borrow starts. That is allowed. What Rust prevents is overlapping read and write access to the same value.

## 11. Borrow Checker Preventing Invalid References

The transcript gives the idea of reading from a vector and then changing the vector. Rust prevents using a reference after the underlying data may have been invalidated.

```rust
fn main() {
    let mut numbers = vec![10, 20, 30];

    let first = &numbers[0];
    println!("first: {first}");

    numbers.clear();
    println!("vector length: {}", numbers.len());
}
```

This works because `first` is no longer used after `numbers.clear()`. If code tried to use `first` after clearing the vector, the compiler would reject it.

## 12. Data Structure Design

The transcript says many Rust beginners struggle less with syntax and more with data modeling. Rust prefers clear ownership instead of casual cyclic references.

A Rust-friendly design may store ownership in one direction:

```rust
struct Page {
    number: u32,
    text: String,
}

struct Book {
    title: String,
    pages: Vec<Page>,
}

fn main() {
    let book = Book {
        title: String::from("Rust Notes"),
        pages: vec![
            Page {
                number: 1,
                text: String::from("Ownership"),
            },
        ],
    };

    println!("{} has {} page(s)", book.title, book.pages.len());
}
```

The `Book` owns the `Page` values. The pages do not point back to the book. This simpler ownership shape is easier for Rust to verify.

## 13. Reference Counting With `Arc`

Sometimes a value really needs multiple owners. The transcript explains reference counting as Rust's controlled way to express that.

```rust
use std::sync::Arc;

fn main() {
    let shared_title = Arc::new(String::from("Rust Notes"));

    let owner1 = Arc::clone(&shared_title);
    let owner2 = Arc::clone(&shared_title);

    println!("{owner1}");
    println!("{owner2}");
}
```

Cloning an `Arc` does not clone the string itself. It creates another owner of the same allocation and increments a counter. The value is cleaned up when the final owner goes away.

## 14. Memory Safety

The transcript presents memory safety as Rust's strongest argument compared with C and C++. Safe Rust prevents bugs such as using freed memory or reading invalid memory.

Example of what Rust prevents:

```rust
fn main() {
    let reference;

    {
        let value = String::from("temporary");
        reference = &value;
    }

    // println!("{reference}"); // error: `value` does not live long enough
}
```

The compiler rejects this because `reference` would point to data that was already cleaned up. This is the kind of invalid memory access Rust is designed to prevent.

## 15. `unsafe`

The transcript explains `unsafe` as an escape hatch for operations the compiler cannot prove safe. It does not disable all Rust checks, and it should be used only when the programmer can manually guarantee the rules.

```rust
fn main() {
    let numbers = vec![10, 20, 30];

    unsafe {
        let second = numbers.get_unchecked(1);
        println!("{second}");
    }
}
```

`get_unchecked` skips the bounds check. This is only valid because index `1` exists. If the programmer uses an invalid index, Rust cannot protect them inside that unsafe operation.

## 16. Safe APIs Over Unsafe Internals

The positive use of `unsafe` in the transcript is building safe abstractions. Unsafe code can exist internally while the public API remains safe.

```rust
struct SafeNumbers {
    values: Vec<i32>,
}

impl SafeNumbers {
    fn new(values: Vec<i32>) -> Self {
        Self { values }
    }

    fn second(&self) -> Option<&i32> {
        self.values.get(1)
    }
}

fn main() {
    let numbers = SafeNumbers::new(vec![10, 20, 30]);

    match numbers.second() {
        Some(value) => println!("second: {value}"),
        None => println!("no second value"),
    }
}
```

This example uses only safe Rust, but it shows the API idea: callers cannot misuse the internals. Real standard-library types may use `unsafe` internally, but expose safe methods like `get`.

## 17. Serialization Shape Checks

The transcript mentions Serde as an example of Rust using types to check incoming data shapes, such as JSON. The key language idea is that data is decoded into a struct with known fields and types.

```rust
struct User {
    id: u32,
    name: String,
}

fn print_user(user: User) {
    println!("{}: {}", user.id, user.name);
}
```

With a library such as Serde, incoming JSON would need to match this structure: `id` must be a number and `name` must be text. The program works with typed data after parsing.

## 18. Editions

Editions let Rust evolve without breaking old code. A crate can choose an edition in `Cargo.toml`.

```toml
[package]
name = "example"
version = "0.1.0"
edition = "2024"
```

The transcript explains editions as Rust's way to make syntax changes while keeping old crates working. Different crates can use different editions and still depend on each other.

## Overall Takeaway

Rust combines low-level control with compiler-enforced discipline. It avoids a garbage collector but still provides memory safety through ownership and borrowing. It avoids implicit nulls and hidden exceptions by making absence and failure part of the type system.

The consistent language philosophy is: make important states explicit, make invalid states hard or impossible to represent, and let the compiler participate in keeping the program correct.
