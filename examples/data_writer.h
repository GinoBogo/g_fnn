// -----------------------------------------------------------------------------
// @file data_writer.h
//
// @date April, 2025
//
// @author Gino Francesco Bogo
// -----------------------------------------------------------------------------

#ifndef DATA_WRITER_H
#define DATA_WRITER_H

#include <stdbool.h> // bool
#include <stdio.h>   // FILE

#include "g_page.h" // f_matrix_t

FILE *data_writer_open(const char *filename);

void data_writer_close(FILE **file);

bool data_writer_next_remark(FILE *file, const char *remark);

bool data_writer_next_values(FILE *file, float *values_ptr, const int values_len);

bool data_writer_next_vector(FILE *file, f_vector_t *vector_ptr);

bool data_writer_next_matrix(FILE *file, f_matrix_t *matrix_ptr);

#endif // DATA_WRITER_H

// -----------------------------------------------------------------------------
// End of File
