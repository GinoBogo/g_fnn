// -----------------------------------------------------------------------------
// @file dataset_reader.h
//
// @date April, 2025
//
// @author Gino Francesco Bogo
// -----------------------------------------------------------------------------

#ifndef DATA_READER_H
#define DATA_READER_H

#include <stdbool.h> // bool

// -----------------------------------------------------------------------------

bool dataset_reader_open(const char *filename);

void dataset_reader_close(void);

bool dataset_reader_next_inputs(float *inputs, const int input_size);

#endif // DATA_READER_H

// -----------------------------------------------------------------------------
// End Of File
