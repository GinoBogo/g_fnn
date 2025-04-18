// -----------------------------------------------------------------------------
// @file main.c
//
// @date November, 2024
//
// @author Gino Francesco Bogo
// -----------------------------------------------------------------------------

#include <libgen.h> // basename
#include <stdio.h>  // FILE, NULL, fprintf, printf, puts
#include <stdlib.h> // atexit, exit
#include <string.h> // strcmp

#include "data_reader.h"
#include "data_writer.h"
#include "g_network.h"

// -----------------------------------------------------------------------------
// Local Definitions
// -----------------------------------------------------------------------------

#define SIZEOF(x) ((int)(sizeof(x) / sizeof(x[0])))

// -----------------------------------------------------------------------------
// Neural Network Layout
// -----------------------------------------------------------------------------

#include "fnn_layout.h"

// -----------------------------------------------------------------------------
// File Handles
// -----------------------------------------------------------------------------

char *fnn_weights_cfg = "fnn_weights.cfg";
char *fnn_dataset_set = "fnn_dataset.set";
char *fnn_outputs_set = "fnn_outputs.set";
char *fnn_weights_out = "fnn_weights.out";
char *fnn_outputs_out = "fnn_outputs.out";

FILE *file_weights_cfg = NULL;
FILE *file_dataset_set = NULL;
FILE *file_outputs_set = NULL;
FILE *file_weights_out = NULL;
FILE *file_outputs_out = NULL;

static void cleanup_resources(void) {
    data_reader_close(&file_weights_cfg);
    data_reader_close(&file_dataset_set);
    data_reader_close(&file_outputs_set);
    data_writer_close(&file_weights_out);
    data_writer_close(&file_outputs_out);
}

// -----------------------------------------------------------------------------
// Network Mode: TRAINING
// -----------------------------------------------------------------------------

static void save_weights_to_file(FILE *file, g_pages_t *pages) {
    if ((file == NULL) || (pages == NULL)) {
        exit(1);
    }

    char remark[32] = {0};
    for (int k = 0; k < pages->len; ++k) {
        snprintf(remark, sizeof(remark), "Layer %d weights", k);

        data_writer_next_remark(file, remark);
        data_writer_next_matrix(file, &pages->ptr[k].w);
    }
}

static void training_mode(g_network_t *network, g_pages_t *pages) {
    // layer 3: actual outputs
    f_vector_t actual_outputs;
    actual_outputs.ptr = &L03_YT[0];
    actual_outputs.len = SIZEOF(L03_YT);

    // weights input file
    file_weights_cfg = data_reader_open(fnn_weights_cfg);
    if (file_weights_cfg == NULL) {
        network->Init_Weights(network, 0.5f);

        file_weights_cfg = data_writer_open(fnn_weights_cfg);
        if (file_weights_cfg == NULL) {
            exit(1);
        }

        save_weights_to_file(file_weights_cfg, pages);
    } else {
        for (int k = 0; k < pages->len; ++k) {
            if (!data_reader_next_matrix(file_weights_cfg, &pages->ptr[k].w)) {
                network->Destroy(network);
                exit(1);
            }
        }
    }

    // outputs file
    file_outputs_set = data_reader_open(fnn_outputs_set);
    if (file_outputs_set == NULL) {
        network->Destroy(network);
        exit(1);
    }

    // weights output file
    file_weights_out = data_writer_open(fnn_weights_out);
    if (file_weights_out == NULL) {
        network->Destroy(network);
        exit(1);
    }

    // load dataset from file
    while (data_reader_next_vector(file_dataset_set, &pages->ptr[0].x)) {
        network->Step_Forward(network);

        // load actual outputs from file
        if (data_reader_next_vector(file_outputs_set, &actual_outputs)) {
            network->Step_Errors(network, &actual_outputs);

            network->Step_Backward(network);
        }

        // save outputs to file
        const int L = pages->len - 1;
        if (!data_writer_next_vector(file_outputs_out, &pages->ptr[L].y)) {
            exit(1);
        }
    }

    save_weights_to_file(file_weights_out, pages);
}

// -----------------------------------------------------------------------------
// Network Mode: INFERENCE
// -----------------------------------------------------------------------------

static void inference_mode(g_network_t *network, g_pages_t *pages) {
    // load weights from file
    file_weights_cfg = data_reader_open(fnn_weights_cfg);
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
// Argument Processing
// -----------------------------------------------------------------------------

static void process_arguments(int argc, char *argv[], bool *is_training) {
    const char *filename = basename(argv[0]);

    if (argc == 1) {
        fprintf(stderr, "Error: No arguments provided\n");
        fprintf(stderr, "For more information use: %s --help\n", filename);
        exit(1);
    }

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];

        if ((strcmp(arg, "--infer") == 0) || (strcmp(arg, "-i") == 0)) {
            *is_training = false;
        }

        else if ((strcmp(arg, "--train") == 0) || (strcmp(arg, "-t") == 0)) {
            *is_training = true;
        }

        else if ((strcmp(arg, "--help") == 0) || (strcmp(arg, "-h") == 0)) {
            // clang-format off
            fprintf(stderr, "Usage:\n");
            fprintf(stderr, "  %s -i [options]\n", filename);
            fprintf(stderr, "  %s -t [options]\n", filename);
            fprintf(stderr, "  %s -h\n", filename);
            fprintf(stderr, "Commands:\n");
            fprintf(stderr, "  -i, --infer               Run in inference mode\n");
            fprintf(stderr, "  -t, --train               Run in training mode\n");
            fprintf(stderr, "  -h, --help                Show this help message\n");
            fprintf(stderr, "Options:\n");
            fprintf(stderr, "  -w, --weights-cfg <file>  The weights cfg file (default: %s)\n", fnn_weights_cfg);
            fprintf(stderr, "  -d, --dataset-set <file>  The dataset set file (default: %s)\n", fnn_dataset_set);
            fprintf(stderr, "  -s, --outputs-set <file>  The outputs set file (default: %s)\n", fnn_outputs_set);
            fprintf(stderr, "  -x, --weights-out <file>  The weights out file (default: %s)\n", fnn_weights_out);
            fprintf(stderr, "  -o, --outputs-out <file>  The outputs out file (default: %s)\n", fnn_outputs_out);
            // clang-format on
            exit(0);
        }

        else if ((strcmp(arg, "--weights-cfg") == 0) || (strcmp(arg, "-w") == 0)) {
            if (i + 1 < argc) {
                fnn_weights_cfg = argv[++i];
            } else {
                fprintf(stderr, "Error: Missing argument for --weights-cfg\n");
                exit(1);
            }
        }

        else if ((strcmp(arg, "--dataset-set") == 0) || (strcmp(arg, "-d") == 0)) {
            if (i + 1 < argc) {
                fnn_dataset_set = argv[++i];
            } else {
                fprintf(stderr, "Error: Missing argument for --dataset-set\n");
                exit(1);
            }
        }

        else if ((strcmp(arg, "--outputs-set") == 0) || (strcmp(arg, "-s") == 0)) {
            if (i + 1 < argc) {
                fnn_outputs_set = argv[++i];
            } else {
                fprintf(stderr, "Error: Missing argument for --outputs-set\n");
                exit(1);
            }
        }

        else if ((strcmp(arg, "--weights-out") == 0) || (strcmp(arg, "-x") == 0)) {
            if (i + 1 < argc) {
                fnn_weights_out = argv[++i];
            } else {
                fprintf(stderr, "Error: Missing argument for --weights-out\n");
                exit(1);
            }
        }

        else if ((strcmp(arg, "--outputs-out") == 0) || (strcmp(arg, "-o") == 0)) {
            if (i + 1 < argc) {
                fnn_outputs_out = argv[++i];
            } else {
                fprintf(stderr, "Error: Missing argument for --outputs-out\n");
                exit(1);
            }
        }

        else {
            fprintf(stderr, "Error: Unknown argument '%s'\n", arg);
            fprintf(stderr, "For more information use: %s --help\n", filename);
            exit(1);
        }
    }
}

// -----------------------------------------------------------------------------
// Main Entry Point
// -----------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    bool is_training = false;

    // process command-line arguments
    process_arguments(argc, argv, &is_training);

    if (is_training) {
        printf("Network mode: training\n");
        printf("  [in ] Weights file: %s\n", fnn_weights_cfg);
        printf("  [in ] Dataset file: %s\n", fnn_dataset_set);
        printf("  [in ] Outputs file: %s\n", fnn_outputs_set);
        printf("  [out] Weights file: %s\n", fnn_weights_out);
        printf("  [out] Outputs file: %s\n", fnn_outputs_out);
    } else {
        printf("Network mode: inference\n");
        printf("  [in ] Weights file: %s\n", fnn_weights_cfg);
        printf("  [in ] Dataset file: %s\n", fnn_dataset_set);
        printf("  [out] Outputs file: %s\n", fnn_outputs_out);
    }

    // register cleanup handler
    atexit(cleanup_resources);

    // -------------------------------------------------------------------------
    // page structure
    // -------------------------------------------------------------------------

    g_page_t page[lho];

    for (int k = 0; k < lho; ++k) {
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
        file_dataset_set = data_reader_open(fnn_dataset_set);
        if (file_dataset_set == NULL) {
            network.Destroy(&network);
            return 1;
        }

        // save outputs to file
        file_outputs_out = data_writer_open(fnn_outputs_out);
        if (file_outputs_out == NULL) {
            network.Destroy(&network);
            return 1;
        }

        if (is_training) {
            training_mode(&network, &pages);
        } else {
            inference_mode(&network, &pages);
        }
        network.Destroy(&network);
    }

    cleanup_resources();

    puts("... Done!");
    return 0;
}

// -----------------------------------------------------------------------------
// End of File
