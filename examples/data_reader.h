// -----------------------------------------------------------------------------
// @file data_reader.h
//
// @date April, 2025
//
// @author Gino Francesco Bogo
// -----------------------------------------------------------------------------

#ifndef DATA_READER_H
#define DATA_READER_H

#include <stdbool.h> // bool
#include <stdio.h>   // FILE

#include "g_page.h" // f_matrix_t

FILE *data_reader_open(const char *filename);

void data_reader_close(FILE **file);

bool data_reader_next_values(FILE *file, float *values_ptr, const int values_len);

bool data_reader_next_matrix(FILE *file, f_matrix_t *matrix_ptr);

#endif // DATA_READER_H

// -----------------------------------------------------------------------------
// End Of File
