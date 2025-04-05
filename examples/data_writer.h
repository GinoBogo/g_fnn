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

FILE *data_writer_open(const char *filename);

void data_writer_close(FILE *file);

bool data_writer_next_values(FILE *file, float *values_ptr, const int values_len);

#endif // DATA_WRITER_H

// -----------------------------------------------------------------------------
// End Of File
