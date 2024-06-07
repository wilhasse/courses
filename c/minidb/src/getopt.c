#include <stdio.h>
#include <stdbool.h>
#include <getopt.h>
#include "file.h"
#include "common.h"

void print_usage(char *argv[]) {
       	printf("Usage %s -n -f <database file>\n",argv[0]);
        printf("\t -n - crate new database file \n");
       	printf("\t -f - (required) path to database file \n");
	return;
}

int main(int argc, char *argv[]) { 
	char *filepath = NULL;
	char *portarg = NULL;
	unsigned short port = 0;
	bool newfile = false;
	bool list = false;
	int c;
	int dbfd = -1;

	while ((c = getopt(argc, argv, "nf:a:l")) != -1) {
		switch (c) {
			case 'n':
				newfile = true;
				break;
			case 'f':
				filepath = optarg;
				break;
			case 'p':
				portarg = optarg;
				break;
			case 'l':
				list = true;
				break;
			case '?':
				printf("Unknown option -%c\n", c);
				break;
			default:
				return -1;

		}
	}

	if (filepath == NULL) {
		printf("Filepath is a reuired argument\n");
		print_usage(argv);
	}

	if (newfile) {
		dbfd = create_db_file(filepath);
		if (dbfd == STATUS_ERROR) {
			printf("Unable to create database file\n");
			return -1;
		}
	} else {
		dbfd = open_db_file(filepath);
		if (dbfd == STATUS_ERROR) {
			printf("Unable to open database file\n");
			return -1;
		}
	}

	printf("Newfile: %d\n" , newfile);
	printf("Filepath: %s\n" , filepath);
}
