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

void g_page_reset(g_page_t *page) {
    if (page != NULL) {
        page->l_id = -1;
        // forward propagation
        page->x.ptr = NULL;
        page->x.len = 0;
        page->w.ptr = NULL;
        page->w.row = 0;
        page->w.col = 0;
        page->z.ptr = NULL;
        page->z.len = 0;
        page->y.ptr = NULL;
        page->y.len = 0;

        // backward propagation
        page->dy_dz.ptr = NULL;
        page->dy_dz.len = 0;
        page->de_dy.ptr = NULL;
        page->de_dy.len = 0;
        page->lr        = 0.0f;
        page->mse       = 0.0f;

        page->af_type     = UNKNOWN;
        page->af_call     = NULL;
        page->af_args.ptr = NULL;
        page->af_args.len = 0;
    }
}

// -----------------------------------------------------------------------------
// End of File
