// -----------------------------------------------------------------------------
// @file data_reader.c
//
// @date April, 2025
//
// @author Gino Francesco Bogo
// -----------------------------------------------------------------------------

#include "data_reader.h"

#include <ctype.h>  // isdigit
#include <errno.h>  // errno
#include <stdio.h>  // FILE, fopen, fclose, fscanf, fgetc, feof, rewind
#include <string.h> // strerror

// -----------------------------------------------------------------------------

static FILE *input_file = NULL;

static void skip_invalid_chars(void) {
    int c;
    while ((c = fgetc(input_file)) != EOF) {
        if (isdigit(c) || c == '.' || c == '-' || c == ',') {
            ungetc(c, input_file);
            break;
        }
    }
}

bool data_reader_open(const char *filename) {
    if (input_file != NULL) {
        data_reader_close();
    }

    if (filename == NULL) {
        printf("[ERROR] Invalid filename\n");
        return false;
    }

    input_file = fopen(filename, "r");
    if (input_file == NULL) {
        printf("[ERROR] Unable to open file '%s': %s\n", filename, strerror(errno));
        return false;
    }

    return true;
}

void data_reader_close(void) {
    if (input_file != NULL) {
        fclose(input_file);
        input_file = NULL;
    }
}

bool data_reader_next_inputs(float *inputs, const int input_size) {
    if (input_file == NULL) {
        printf("[ERROR] No file open\n");
        return false;
    }

    if (inputs == NULL || input_size <= 0) {
        printf("[ERROR] Invalid arguments for next inputs\n");
        return false;
    }

    skip_invalid_chars();

    int items_read = fscanf(input_file,
                            "%f,%f,%f,%f,%f,%f,%f",
                            &inputs[0],
                            &inputs[1],
                            &inputs[2],
                            &inputs[3],
                            &inputs[4],
                            &inputs[5],
                            &inputs[6]);

    if (items_read != input_size) {
        if (!feof(input_file)) {
            printf("[ERROR] Invalid input format, expected %d values per line\n", input_size);
            return false;
        }
    }

    int c;
    while ((c = fgetc(input_file)) != EOF && c != '\n') {
        // Do nothing, just consume the characters
    }

    return true;
}

// -----------------------------------------------------------------------------
// End Of File
