#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <sys/stat.h>
#include "file.h"

int generate_file() {

    struct stat filest;

    // open file
    int fd = open("./test_file.bin", O_RDWR | O_CREAT, 0644);
    if (fd == -1) {
       perror("open");
       return -1;
    }

    // size
    fstat(fd,&filest);
    printf("File Size %lu\n",filest.st_size);

    // write
    char *a_buf = "some test data\n";
    write(fd, a_buf, strlen(a_buf));

    // new size
    fstat(fd,&filest);
    printf("File Wrote\n");
    printf("File Size %lu\n",filest.st_size);

    // close
    close(fd);

    return 0;
}
