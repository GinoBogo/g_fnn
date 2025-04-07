// -----------------------------------------------------------------------------
// @file g_page.c
//
// @date April, 2025
//
// @author Gino Francesco Bogo
// -----------------------------------------------------------------------------

#include "g_page.h"

#include <stdbool.h> // bool
#include <stddef.h>  // NULL

// -----------------------------------------------------------------------------

float *f_matrix_row(f_matrix_t *mat, int row) {
    float *rvalue = NULL;

    if ((mat != NULL) && (mat->ptr != NULL)) {
        const bool chk_1 = row >= 0;
        const bool chk_2 = mat->row > row;
        const bool chk_3 = mat->col > 0;

        if (chk_1 && chk_2 && chk_3) {
            rvalue = mat->ptr + (row * mat->col);
        }
    }

    return rvalue;
}

float *f_matrix_at(f_matrix_t *mat, int row, int col) {
    float *rvalue = NULL;

    if ((mat != NULL) && (mat->ptr != NULL)) {
        const bool chk_1 = row >= 0 && col >= 0;
        const bool chk_2 = mat->row > row;
        const bool chk_3 = mat->col > col;

        if (chk_1 && chk_2 && chk_3) {
            rvalue = mat->ptr + (row * mat->col + col);
        }
    }

    return rvalue;
}

// -----------------------------------------------------------------------------
// End Of File
