// -----------------------------------------------------------------------------
// @file data_writer.c
//
// @date April, 2025
//
// @author Gino Francesco Bogo
// -----------------------------------------------------------------------------

#include "data_writer.h"

#include <errno.h>  // errno
#include <stdio.h>  // FILE, fopen, fclose, fprintf
#include <string.h> // strerror

// -----------------------------------------------------------------------------

FILE *data_writer_open(const char *filename) {
    if (filename == NULL) {
        printf("[ERROR] Invalid filename\n");
        return NULL;
    }

    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("[ERROR] Unable to open file '%s': %s\n", filename, strerror(errno));
        return NULL;
    }

    return file;
}

void data_writer_close(FILE **file) {
    if (*file != NULL) {
        fclose(*file);
        *file = NULL;
    }
}

bool data_writer_next_values(FILE *file, float *values_ptr, const int values_len) {
    if (file == NULL) {
        printf("[ERROR] No file open\n");
        return false;
    }

    if (values_ptr == NULL || values_len <= 0) {
        printf("[ERROR] Invalid arguments for next values\n");
        return false;
    }

    for (int i = 0; i < values_len; i++) {
        if (fprintf(file, "%14.6e", values_ptr[i]) < 0) {
            return false;
        }
        if (i < values_len - 1) {
            if (fprintf(file, ",") < 0) {
                return false;
            }
        }
    }

    if (fprintf(file, "\n") < 0) {
        return false;
    }

    return true;
}

// -----------------------------------------------------------------------------
// End Of File
