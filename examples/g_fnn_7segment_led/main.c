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

#define TRAINING_MODE 1

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
float             L01_LR                       = 0.01f;
g_act_func_type_t L01_AF_TYPE                  = LEAKY_RELU;
float             L01_AF_ARGS[1]               = {0.01f};

// layer 2: hidden layer
float             L02_W[20][SIZEOF(L01_Y) + 1] = {{0.0f}};
float             L02_Z[SIZEOF(L02_W)]         = {0.0f};
float             L02_Y[SIZEOF(L02_Z)]         = {0.0f};
float             L02_dY_dZ[SIZEOF(L02_Y)]     = {0.0f};
float             L02_dE_dY[SIZEOF(L02_Y)]     = {0.0f};
float             L02_LR                       = 0.02f;
g_act_func_type_t L02_AF_TYPE                  = LEAKY_RELU;
float             L02_AF_ARGS[1]               = {0.01f};

// layer 3: output layer
float             L03_W[10][SIZEOF(L02_Y) + 1] = {{0.0f}};
float             L03_Z[SIZEOF(L03_W)]         = {0.0f};
float             L03_Y[SIZEOF(L03_Z)]         = {0.0f};
float             L03_dY_dZ[SIZEOF(L03_Y)]     = {0.0f};
float             L03_dE_dY[SIZEOF(L03_Y)]     = {0.0f};
float             L03_LR                       = 0.03f;
g_act_func_type_t L03_AF_TYPE                  = SIGMOID;
float             L03_AF_ARGS[1]               = {0.0f};

// layer 3: actual outputs (Y target)
float L03_YT[SIZEOF(L03_Y)] = {0.0f};

// -----------------------------------------------------------------------------
// File Handles
// -----------------------------------------------------------------------------

FILE *file_weights_cfg = NULL;
FILE *file_dataset_set = NULL;
FILE *file_outputs_act = NULL;
FILE *file_weights_out = NULL;
FILE *file_outputs_out = NULL;

static void cleanup_resources(void) {
    data_reader_close(&file_weights_cfg);
    data_reader_close(&file_dataset_set);
    data_reader_close(&file_outputs_act);
    data_writer_close(&file_weights_out);
    data_writer_close(&file_outputs_out);
}

// -----------------------------------------------------------------------------
// Network Mode: TRAINING
// -----------------------------------------------------------------------------

void training_mode(g_network_t *network, g_pages_t *pages) {
    // layer 3: actual outputs
    f_vector_t actual_outputs;
    actual_outputs.ptr = &L03_YT[0];
    actual_outputs.len = SIZEOF(L03_YT);

    network->Init_Weights(network, 0.5f);

    file_outputs_act = data_reader_open("fnn_outputs.act");
    if (file_outputs_act == NULL) {
        network->Destroy(network);
        exit(1);
    }

    file_weights_out = data_writer_open("fnn_weights.out");
    if (file_weights_out == NULL) {
        network->Destroy(network);
        exit(1);
    }

    // load dataset from file
    while (data_reader_next_vector(file_dataset_set, &pages->ptr[0].x)) {
        network->Step_Forward(network);

        // load actual outputs from file
        if (data_reader_next_vector(file_outputs_act, &actual_outputs)) {
            network->Step_Errors(network, &actual_outputs);

            network->Step_Backward(network);
        }

        // save outputs to file
        const int L = pages->len - 1;
        if (!data_writer_next_vector(file_outputs_out, &pages->ptr[L].y)) {
            exit(1);
        }
    }

    // save weights to file
    data_writer_next_remark(file_weights_out, "Layer 1 weights");
    data_writer_next_matrix(file_weights_out, &pages->ptr[0].w);
    data_writer_next_remark(file_weights_out, "Layer 2 weights");
    data_writer_next_matrix(file_weights_out, &pages->ptr[1].w);
    data_writer_next_remark(file_weights_out, "Layer 3 weights");
    data_writer_next_matrix(file_weights_out, &pages->ptr[2].w);
}

// -----------------------------------------------------------------------------
// Network Mode: INFERENCE
// -----------------------------------------------------------------------------

void inference_mode(g_network_t *network, g_pages_t *pages) {
    // load weights from file
    file_weights_cfg = data_reader_open("fnn_weights.cfg");
    if (file_weights_cfg == NULL) {
        network->Destroy(network);
        exit(1);
    }

    for (int k = 0; k < pages->len; ++k) {
        if (!data_reader_next_matrix(file_weights_cfg, &pages->ptr[k].w)) {
            network->Destroy(network);
            exit(1);
        }
    }
    data_reader_close(&file_weights_cfg);

    // load dataset from file
    while (data_reader_next_vector(file_dataset_set, &pages->ptr[0].x)) {
        network->Step_Forward(network);

        // save outputs to file
        const int L = pages->len - 1;
        if (!data_writer_next_vector(file_outputs_out, &pages->ptr[L].y)) {
            network->Destroy(network);
            exit(1);
        }
    }
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
    // page structure
    // -------------------------------------------------------------------------

    g_page_t page[3];

    for (int k = 0; k < SIZEOF(page); ++k) {
        g_layer_page_reset(&page[k]);
        page[k].l_id = k;
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
        // load dataset from file
        file_dataset_set = data_reader_open("fnn_dataset.set");
        if (file_dataset_set == NULL) {
            network.Destroy(&network);
            return 1;
        }

        // save outputs to file
        file_outputs_out = data_writer_open("fnn_outputs.out");
        if (file_outputs_out == NULL) {
            network.Destroy(&network);
            return 1;
        }

#if TRAINING_MODE
        training_mode(&network, &pages);
#else
        inference_mode(&network, &pages);
#endif
        network.Destroy(&network);
    }

    cleanup_resources();

    puts("... Done!");
    return 0;
}

// -----------------------------------------------------------------------------
// End Of File
