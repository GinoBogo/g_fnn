// -----------------------------------------------------------------------------
// @file fnn_layout.h
//
// @date April, 2025
//
// @author Gino Francesco Bogo
// -----------------------------------------------------------------------------

#ifndef FNN_LAYOUT_H
#define FNN_LAYOUT_H

#include <g_page.h>

enum fnn_layout {
    l00 = 7,  // neurons in layer 0 (input)
    l01 = 20, // neurons in layer 1 (hidden)
    l02 = 20, // neurons in layer 2 (hidden)
    l03 = 10, // neurons in layer 3 (output)
    lho = 3   // number of hidden/output layers
};

// layer 0: virtual input layer
float L00_Y[l00] = {0.0f};

// layer 1: hidden layer
float             L01_W[l01][l00 + 1] = {{0.0f}};
float             L01_Z[l01]          = {0.0f};
float             L01_Y[l01]          = {0.0f};
float             L01_dY_dZ[l01]      = {0.0f};
float             L01_dE_dY[l01]      = {0.0f};
float             L01_LR              = 0.01f;
g_act_func_type_t L01_AF_TYPE         = LEAKY_RELU;
float             L01_AF_ARGS[1]      = {0.01f};

// layer 2: hidden layer
float             L02_W[l02][l01 + 1] = {{0.0f}};
float             L02_Z[l02]          = {0.0f};
float             L02_Y[l02]          = {0.0f};
float             L02_dY_dZ[l02]      = {0.0f};
float             L02_dE_dY[l02]      = {0.0f};
float             L02_LR              = 0.02f;
g_act_func_type_t L02_AF_TYPE         = LEAKY_RELU;
float             L02_AF_ARGS[1]      = {0.01f};

// layer 3: output layer
float             L03_W[l03][l02 + 1] = {{0.0f}};
float             L03_Z[l03]          = {0.0f};
float             L03_Y[l03]          = {0.0f};
float             L03_dY_dZ[l03]      = {0.0f};
float             L03_dE_dY[l03]      = {0.0f};
float             L03_LR              = 0.03f;
g_act_func_type_t L03_AF_TYPE         = SIGMOID;
float             L03_AF_ARGS[1]      = {0.0f};

// layer 3: actual outputs (Y target)
float L03_YT[l03] = {0.0f};

#endif // FNN_LAYOUT_H

// -----------------------------------------------------------------------------
// End of File
