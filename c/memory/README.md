# Bootdev Course Exercises 🎓

Repository containing exercises solved during Bootdev Course using the playground environment. Claude 3.5 Sonnet (Anthropic AI) assisted in generating utility classes and Makefiles.

## Project Structure 🗂️

### Core Utilities
- **munit/** - Testing framework that asserts exercise results, providing detailed feedback for failures and successes
- **bootlib/** - Memory leak detector that wraps malloc functionality

## Tools Used 🛠️
- Bootdev Course Playground
- Claude 3.5 Sonnet for utility code generation
- C Programming Language

## Note 📝
This repository serves as documentation of my learning journey through the Bootdev Course, highlighting both the exercises completed and the tools utilized along the way.

# Exercises

- [munit](munit) -> Helper for all unit test assert using different types 
- [bootlib](bootlib) -> Units to verify if malloc produced any leak
- [unittest](unittest) -> How to use an unittest example
- [struct](struct) -> Simple struct example
- [structp](structp) -> Passing struct to a function (copy vs in-place)
- [pointer](pointer) -> Using pointer to a struct
- [flatp](flatp) -> Iterating over an array of struct using a flat pointer
- [string](string) -> String helper functions, using lib
- [mstruct](mstruct) -> Forward, Mutual Struct and typedef
- [union](union) -> Simple Union example
- [union_helper](union_helper) -> Union example with Helper Fields
- [stack](stack) -> Example showing problem using stack memory, different ways to solve it
- [malloc](malloc) -> Using malloc to initialize an array
- [ppointer](ppointer) -> Why do you need pointer to another pointer? Correct and Incorrect usage
- [token](token) -> Array of pointers to string (pointer to an char array)
- [void_pointer](void_pointer) -> Using void pointer to point to different types using switch. Elegant solution in [Zig](../../zig/zig-exercises/void_pointer.zig) using comptime
- [generic_swap](generic_swap) -> Swapping two values with generic types using pointer
- [my_stack](my_stack) -> Implementing a simple stack data structure using void pointer
- [object](object) -> Complete exercise using structs to hold values arrays of different type
- [gc_obj](gc_obj) -> Implementing garbage collector in the snekobjects
- [sweep](sweep) -> Implementing garbage collector Mark and Sweep to handle cycles while referencing objects
