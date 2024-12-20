# Introduction

Software for generating documentation and call graph  

Install Doxygen  
https://www.doxygen.nl/  

Install Graphziz  
https://graphviz.org/download/

Note: Graphziz select Add Graphviz to the system PATH 

# Config

Generate template config

```bash
doxygen -g Doxyfile
```

Edit config to customize. Default

```ini
# Set the input directory
INPUT                  = D:\polardbx\polardbx-sql

# Enable recursive directory scan
RECURSIVE              = YES

# Extract everything
EXTRACT_ALL            = YES

# Enable Graphviz support
HAVE_DOT              = YES
CLASS_DIAGRAMS        = YES
HIDE_UNDOC_RELATIONS  = NO
CALL_GRAPH            = YES
CALLER_GRAPH          = YES
COLLABORATION_GRAPH   = YES
UML_LOOK              = YES
GENERATE_HTML        = YES
GENERATE_LATEX       = NO

# Set output directory
OUTPUT_DIRECTORY       = D:\polardbx\polardbx-sql\doxygen-docs

# Generate comprehensive class diagrams
GENERATE_TREEVIEW      = YES
```

Java

```ini
# Java specific settings
FILE_PATTERNS         = *.java *.xml *.properties
OPTIMIZE_JAVA_OUTPUT  = YES
EXTRACT_ALL          = YES
EXTRACT_PRIVATE      = YES
EXTRACT_PACKAGE      = YES
EXTRACT_STATIC       = YES
EXTRACT_LOCAL_CLASSES = YES
```

# Run

```bash
doxygen Doxyfile
```
