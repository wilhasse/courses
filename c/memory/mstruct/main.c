#include "munit.h"
#include "exercise.h"
#include <string.h>

munit_case(RUN, test_assign_employee1, {
  employee_t emp = create_employee(2, "CEO Dax");
  department_t dept = create_department("C Suite");
  assign_employee(&emp, &dept);
  munit_assert_string_equal(emp.department->name, "C Suite", "should match names");
});

munit_case(RUN, test_assign_manager1, {
  employee_t manager = create_employee(3, "Influencer Prime");
  department_t dept = create_department("Marketing");
  assign_manager(&dept, &manager);
  munit_assert_string_equal(dept.manager->name, "Influencer Prime", "should match names");
});

munit_case(SUBMIT, test_assign_employee2, {
  employee_t emp = create_employee(4, "Vegan Intern Adam");
  department_t dept = create_department("Marketing");
  assign_employee(&emp, &dept);
  munit_assert_string_equal(emp.department->name, "Marketing", "should match names");
});

munit_case(SUBMIT, test_assign_manager2, {
  employee_t manager = create_employee(5, "CDO David");
  department_t dept = create_department("C Suite");
  assign_manager(&dept, &manager);
  munit_assert_string_equal(dept.manager->name, "CDO David", "should match names");
  munit_assert_int(manager.id, ==, 5, "should match ids");
});

int main() {
  MunitTest tests[] = {
    munit_test("/test_assign_employee1", test_assign_employee1),
    munit_test("/test_assign_manager1", test_assign_manager1),
    munit_test("/test_assign_employee2", test_assign_employee2),
    munit_test("/test_assign_manager2", test_assign_manager2),
    munit_null_test,
  };

  MunitSuite suite = munit_suite("employee_department_tests", tests);

  return munit_suite_main(&suite, NULL, 0, NULL);
}
