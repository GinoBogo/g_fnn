// -----------------------------------------------------------------------------
// @file main.c
//
// @date November, 2024
//
// @author Gino Francesco Bogo
// -----------------------------------------------------------------------------

#include <stdio.h> // puts

#include "g_network.h"

#define SIZEOF(x) (sizeof(x) / sizeof(x[0]))

// -----------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    float L00_X[7];

    float             L00_W[20][8];
    float             L00_Z[20];
    float             L00_Y[20];
    g_act_func_type_t L00_AF_TYPE    = LEAKY_RELU;
    float             L00_AF_ARGS[1] = {0.001};

    float             L01_W[20][21];
    float             L01_Z[20];
    float             L01_Y[20];
    g_act_func_type_t L01_AF_TYPE    = LEAKY_RELU;
    float             L01_AF_ARGS[1] = {0.001};

    float             L02_W[10][21];
    float             L02_Z[10];
    float             L02_Y[10];
    g_act_func_type_t L02_AF_TYPE    = SIGMOID;
    float             L02_AF_ARGS[1] = {0.0};

    g_layer_data_t layer_data[3];

    for (int i = 0; i < 3; ++i) {
        g_layer_data_reset(&layer_data[i]);
    }

    layer_data[0].x.ptr       = L00_X;
    layer_data[0].x.len       = SIZEOF(L00_X);
    layer_data[0].w.ptr       = (float *)L00_W;
    layer_data[0].w.row       = SIZEOF(L00_W);
    layer_data[0].w.col       = SIZEOF(L00_W[0]);
    layer_data[0].z.ptr       = L00_Z;
    layer_data[0].z.len       = SIZEOF(L00_Z);
    layer_data[0].y.ptr       = L00_Y;
    layer_data[0].y.len       = SIZEOF(L00_Y);
    layer_data[0].af_type     = L00_AF_TYPE;
    layer_data[0].af_args.ptr = L00_AF_ARGS;
    layer_data[0].af_args.len = SIZEOF(L00_AF_ARGS);

    layer_data[1].x.ptr       = L00_Y;
    layer_data[1].x.len       = SIZEOF(L00_Y);
    layer_data[1].w.ptr       = (float *)L01_W;
    layer_data[1].w.row       = SIZEOF(L01_W);
    layer_data[1].w.col       = SIZEOF(L01_W[0]);
    layer_data[1].z.ptr       = L01_Z;
    layer_data[1].z.len       = SIZEOF(L01_Z);
    layer_data[1].y.ptr       = L01_Y;
    layer_data[1].y.len       = SIZEOF(L01_Y);
    layer_data[1].af_type     = L01_AF_TYPE;
    layer_data[1].af_args.ptr = L01_AF_ARGS;
    layer_data[1].af_args.len = SIZEOF(L01_AF_ARGS);

    layer_data[2].x.ptr       = L01_Y;
    layer_data[2].x.len       = SIZEOF(L01_Y);
    layer_data[2].w.ptr       = (float *)L02_W;
    layer_data[2].w.row       = SIZEOF(L02_W);
    layer_data[2].w.col       = SIZEOF(L02_W[0]);
    layer_data[2].z.ptr       = L02_Z;
    layer_data[2].z.len       = SIZEOF(L02_Z);
    layer_data[2].y.ptr       = L02_Y;
    layer_data[2].y.len       = SIZEOF(L02_Y);
    layer_data[2].af_type     = L02_AF_TYPE;
    layer_data[2].af_args.ptr = L02_AF_ARGS;
    layer_data[2].af_args.len = SIZEOF(L02_AF_ARGS);

    g_layers_data_t layers_data;

    layers_data.ptr = layer_data;
    layers_data.len = SIZEOF(layer_data);

    g_network_t network;

    g_network_link(&network);

    if (network.Create(&network, &layers_data)) {
        network.Step_Fwd(&network);
    }

    network.Destroy(&network);

    puts("... Done!");
    return 0;
}

// -----------------------------------------------------------------------------
// End Of File
