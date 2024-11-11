#include <stdlib.h>

#include "exercise.h"

token_t** create_token_pointer_array(token_t* tokens, size_t count) {
  token_t** token_pointers = malloc(count * sizeof(token_t*));
  if (token_pointers == NULL) {
    exit(1);
  }
  for (size_t i = 0; i < count; ++i) {
    token_t* token = malloc(sizeof(token_t*));
    token->literal = tokens[i].literal;
    token->line = tokens[i].line;
    token->column = tokens[i].column;
    token_pointers[i] = token;
  }
  return token_pointers;
}
