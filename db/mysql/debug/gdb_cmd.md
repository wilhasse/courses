# Commands

Run gdb

```bash
# run
gdb binary

# run with args
gdb --args ./runtime_output_directory/decrypt "~/keyring-encrypted" "~/ACESSO_c.ibd"

# or set args inside gdb
gdb ./runtime_output_directory/decrypt
(gdb) set args ~/keyring-encrypted ~/ACESSO_c.ibd
```

Debugging

```bash
# watch a variable, it will trigger when it changes value
(gdb) watch *system_charset_info

# add to a function
(gdb) break main

# run
(gdb) run
```

Inspecting

```bash
# Step over
(gdb) next 

# Step into
(gdb) step    

# See variable
(gdb) print cs_arg

# Show backtrace to see where we are
(gdb) bt

# Local variables
(gdb) info locals
```

