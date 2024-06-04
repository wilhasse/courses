#include <stdio.h>
#include "struct.h"

int generate_struct() {

    struct Employee {
      char *name;
      int age;
    };

    //
    struct Employee employee;
    employee.name = "Steve";
    employee.age = 42;

    printf("%s, aged %d years\n", employee.name , employee.age);
    return 0;
}
