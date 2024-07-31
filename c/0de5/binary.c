#include <stdio.h>

int main() {
  unsigned char bytes[6] = {};

  // 5 bytes + string terminated
  bytes[0] = 0b01001000;
  bytes[1] = 0x65;
  bytes[2] = 108;
  bytes[3] = 0154;
  bytes[4] = 'o';
  bytes[5] = 0;

  for (int i = 0;i < 6; i++) {

    // index - dec - hex
    printf("[%2u] %3u %2x ", bytes[i], bytes[i], bytes[i]);

    // binary
    for (int b = 7; b>=0; b--) {

      // expand binary numbers
      int bit = (bytes[i] >> b) & 0b0000001;
      printf("%u", bit);
    }

    // caracter
    printf(" %c\n",bytes[i]);
  }

  // entire string
  printf(" '%s'\n",bytes);
}
