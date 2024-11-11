void swap_strings(char **a, char **b) {
  // ?
  char *swap;
  swap = *b;
  *b = *a;
  *a = swap;
}
