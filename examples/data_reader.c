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

static void __skip_invalid_chars(FILE *file) {
    int c;
    while ((c = fgetc(file)) != EOF) {
        // Skip lines that start with '#'
        if (c == '#') {
            while ((c = fgetc(file)) != EOF && c != '\n') {
                // Consume the rest of the line
            }
            continue; // Check the next character after the comment line
        }

        // Check for valid characters
        if (isdigit(c)     //
            || c == '.'    //
            || c == '-'    //
            || c == ','    //
            || c == 'e'    //
            || c == 'E') { //
            ungetc(c, file);
            break;
        }
    }
}

FILE *data_reader_open(const char *filename) {
    if (filename == NULL) {
        printf("[ERROR] Invalid filename\n");
        return NULL;
    }

    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("[ERROR] Unable to open file '%s': %s\n", filename, strerror(errno));
        return NULL;
    }

    return file;
}

void data_reader_close(FILE **file) {
    if (*file != NULL) {
        fclose(*file);
        *file = NULL;
    }
}

bool data_reader_next_values(FILE *file, float *values_ptr, const int values_len) {
    if (file == NULL) {
        printf("[ERROR] No file open\n");
        return false;
    }

    if (values_ptr == NULL || values_len <= 0) {
        printf("[ERROR] Invalid arguments for next values\n");
        return false;
    }

    __skip_invalid_chars(file);

    int items_read = 0;
    for (int i = 0; i < values_len; i++) {
        if (fscanf(file, "%f", &values_ptr[i]) != 1) {
            break;
        }
        items_read++;
        if (i < values_len - 1) {
            int c = fgetc(file);
            if (c != ',' && c != EOF) {
                ungetc(c, file);
            }
        }
    }

    if (items_read != values_len) {
        if (!feof(file)) {
            printf("[ERROR] Invalid input format, expected %d values per line\n", values_len);
        }
        return false;
    }

    int c;
    while ((c = fgetc(file)) != EOF && c != '\n') {
        // Do nothing, just consume the characters
    }

    return true;
}

bool data_reader_next_vector(FILE *file, f_vector_t *vector_ptr) {
    if (file == NULL) {
        printf("[ERROR] No file open\n");
        return false;
    }

    if (vector_ptr == NULL || vector_ptr->len <= 0) {
        printf("[ERROR] Invalid arguments for next vector\n");
        return false;
    }

    return data_reader_next_values(file, vector_ptr->ptr, vector_ptr->len);
}

bool data_reader_next_matrix(FILE *file, f_matrix_t *matrix_ptr) {
    if (file == NULL) {
        printf("[ERROR] No file open\n");
        return false;
    }

    if (matrix_ptr == NULL || matrix_ptr->row <= 0 || matrix_ptr->col <= 0) {
        printf("[ERROR] Invalid arguments for next matrix\n");
        return false;
    }

    for (int i = 0; i < matrix_ptr->row; i++) {
        if (!data_reader_next_values(file, f_matrix_row(matrix_ptr, i), matrix_ptr->col)) {
            return false;
        }
    }

    return true;
}

// -----------------------------------------------------------------------------
// End of File
