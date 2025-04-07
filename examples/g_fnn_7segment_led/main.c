// -----------------------------------------------------------------------------
// @file main.c
//
// @date November, 2024
//
// @author Gino Francesco Bogo
// -----------------------------------------------------------------------------

#include <stdio.h>  // puts
#include <stdlib.h> // atexit
//
#include "data_reader.h"
#include "data_writer.h"
#include "g_network.h"

#define SIZEOF(x) ((int)(sizeof(x) / sizeof(x[0])))

// -----------------------------------------------------------------------------

FILE *network_inputs_file  = NULL;
FILE *actual_outputs_file  = NULL;
FILE *network_outputs_file = NULL;

static void cleanup_resources(void) {
    data_reader_close(&network_inputs_file);
    data_reader_close(&actual_outputs_file);
    data_writer_close(&network_outputs_file);
}

// -----------------------------------------------------------------------------
// Entry Point
// -----------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    // register cleanup handler
    atexit(cleanup_resources);

    // -------------------------------------------------------------------------
    // network topology definition
    // -------------------------------------------------------------------------

    // layer 0: virtual input layer (store outputs only)
    float L00_Y[7] = {0.0f};

    // layer 1: hidden layer
    float             L01_W[20][SIZEOF(L00_Y) + 1] = {{0.0f}};
    float             L01_Z[SIZEOF(L01_W)]         = {0.0f};
    float             L01_Y[SIZEOF(L01_Z)]         = {0.0f};
    float             L01_dY_dZ[SIZEOF(L01_Y)]     = {0.0f};
    float             L01_dE_dY[SIZEOF(L01_Y)]     = {0.0f};
    float             L01_LR                       = 0.20f;
    g_act_func_type_t L01_AF_TYPE                  = LEAKY_RELU;
    float             L01_AF_ARGS[1]               = {0.01f};

    // layer 2: hidden layer
    float             L02_W[20][SIZEOF(L01_Y) + 1] = {{0.0f}};
    float             L02_Z[SIZEOF(L02_W)]         = {0.0f};
    float             L02_Y[SIZEOF(L02_Z)]         = {0.0f};
    float             L02_dY_dZ[SIZEOF(L02_Y)]     = {0.0f};
    float             L02_dE_dY[SIZEOF(L02_Y)]     = {0.0f};
    float             L02_LR                       = 0.10f;
    g_act_func_type_t L02_AF_TYPE                  = LEAKY_RELU;
    float             L02_AF_ARGS[1]               = {0.01f};

    // layer 3: output layer
    float             L03_W[10][SIZEOF(L02_Y) + 1] = {{0.0f}};
    float             L03_Z[SIZEOF(L03_W)]         = {0.0f};
    float             L03_Y[SIZEOF(L03_Z)]         = {0.0f};
    float             L03_dY_dZ[SIZEOF(L03_Y)]     = {0.0f};
    float             L03_dE_dY[SIZEOF(L03_Y)]     = {0.0f};
    float             L03_LR                       = 0.05f;
    g_act_func_type_t L03_AF_TYPE                  = SIGMOID;
    float             L03_AF_ARGS[1]               = {0.0f};

    // layer 3: actual outputs (Y target)
    float L03_YT[SIZEOF(L03_Y)] = {0.0f};

    // -------------------------------------------------------------------------
    // page structure
    // -------------------------------------------------------------------------
    g_page_t page[3];

    for (int i = 0; i < SIZEOF(page); ++i) {
        g_layer_page_reset(&page[i]);
        page[i].l_id = i;
    }

    // layer 1: hidden layer
    page[0].x.ptr       = &L00_Y[0];
    page[0].x.len       = SIZEOF(L00_Y);
    page[0].w.ptr       = &L01_W[0][0];
    page[0].w.row       = SIZEOF(L01_W);
    page[0].w.col       = SIZEOF(L01_W[0]);
    page[0].z.ptr       = &L01_Z[0];
    page[0].z.len       = SIZEOF(L01_Z);
    page[0].y.ptr       = &L01_Y[0];
    page[0].y.len       = SIZEOF(L01_Y);
    page[0].dy_dz.ptr   = &L01_dY_dZ[0];
    page[0].dy_dz.len   = SIZEOF(L01_dY_dZ);
    page[0].de_dy.ptr   = &L01_dE_dY[0];
    page[0].de_dy.len   = SIZEOF(L01_dE_dY);
    page[0].lr          = L01_LR;
    page[0].af_type     = L01_AF_TYPE;
    page[0].af_args.ptr = L01_AF_ARGS;
    page[0].af_args.len = SIZEOF(L01_AF_ARGS);

    // layer 2: hidden layer
    page[1].x.ptr       = &L01_Y[0];
    page[1].x.len       = SIZEOF(L01_Y);
    page[1].w.ptr       = &L02_W[0][0];
    page[1].w.row       = SIZEOF(L02_W);
    page[1].w.col       = SIZEOF(L02_W[0]);
    page[1].z.ptr       = &L02_Z[0];
    page[1].z.len       = SIZEOF(L02_Z);
    page[1].y.ptr       = &L02_Y[0];
    page[1].y.len       = SIZEOF(L02_Y);
    page[1].dy_dz.ptr   = &L02_dY_dZ[0];
    page[1].dy_dz.len   = SIZEOF(L02_dY_dZ);
    page[1].de_dy.ptr   = &L02_dE_dY[0];
    page[1].de_dy.len   = SIZEOF(L02_dE_dY);
    page[1].lr          = L02_LR;
    page[1].af_type     = L02_AF_TYPE;
    page[1].af_args.ptr = L02_AF_ARGS;
    page[1].af_args.len = SIZEOF(L02_AF_ARGS);

    // layer 3: output layer
    page[2].x.ptr       = &L02_Y[0];
    page[2].x.len       = SIZEOF(L02_Y);
    page[2].w.ptr       = &L03_W[0][0];
    page[2].w.row       = SIZEOF(L03_W);
    page[2].w.col       = SIZEOF(L03_W[0]);
    page[2].z.ptr       = &L03_Z[0];
    page[2].z.len       = SIZEOF(L03_Z);
    page[2].y.ptr       = &L03_Y[0];
    page[2].y.len       = SIZEOF(L03_Y);
    page[2].dy_dz.ptr   = &L03_dY_dZ[0];
    page[2].dy_dz.len   = SIZEOF(L03_dY_dZ);
    page[2].de_dy.ptr   = &L03_dE_dY[0];
    page[2].de_dy.len   = SIZEOF(L03_dE_dY);
    page[2].lr          = L03_LR;
    page[2].af_type     = L03_AF_TYPE;
    page[2].af_args.ptr = L03_AF_ARGS;
    page[2].af_args.len = SIZEOF(L03_AF_ARGS);

    // layer 3: actual outputs
    f_vector_t actual_outputs;
    actual_outputs.ptr = &L03_YT[0];
    actual_outputs.len = SIZEOF(L03_YT);

    // -------------------------------------------------------------------------
    // pages structure
    // -------------------------------------------------------------------------
    g_pages_t pages;

    pages.ptr = &page[0];
    pages.len = SIZEOF(page);

    // -------------------------------------------------------------------------
    // network structure
    // -------------------------------------------------------------------------
    g_network_t network;

    g_network_link(&network);

    if (network.Create(&network, &pages)) {
        network.Init_Weights(&network);

        // network inputs stream
        network_inputs_file = data_reader_open("network_inputs.txt");
        if (network_inputs_file == NULL) {
            network.Destroy(&network);
            return 1;
        }

        // actual outputs stream
        actual_outputs_file = data_reader_open("actual_outputs.txt");
        if (actual_outputs_file == NULL) {
            data_reader_close(&network_inputs_file);
            network.Destroy(&network);
            return 1;
        }

        // network outputs stream
        network_outputs_file = data_writer_open("network_outputs.txt");
        if (network_outputs_file == NULL) {
            data_reader_close(&network_inputs_file);
            data_reader_close(&actual_outputs_file);
            network.Destroy(&network);
            return 1;
        }

        while (data_reader_next_values(network_inputs_file, L00_Y, SIZEOF(L00_Y))) {
            network.Step_Forward(&network);

            if (data_reader_next_values(actual_outputs_file, L03_YT, SIZEOF(L03_YT))) {
                network.Step_Errors(&network, &actual_outputs);

                network.Step_Backward(&network);
            }

            if (!data_writer_next_values(network_outputs_file, L03_Y, SIZEOF(L03_Y))) {
                break;
            }
        }

        data_reader_close(&network_inputs_file);
        data_reader_close(&actual_outputs_file);
        data_writer_close(&network_outputs_file);
    }

    network.Destroy(&network);

    puts("... Done!");
    return 0;
}

// -----------------------------------------------------------------------------
// End Of File
