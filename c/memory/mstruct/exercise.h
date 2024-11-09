#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ?
typedef struct Employee employee_t;
typedef struct Department department_t;

struct Employee {
  // ?
  int id;
  char *name;
  department_t *department;
};

struct Department {
    // ?
    char *name;
    employee_t *manager;
};

employee_t create_employee(int id, char *name);
department_t create_department(char *name);

void assign_employee(employee_t *emp, department_t *department);
void assign_manager(department_t *dept, employee_t *manager);
