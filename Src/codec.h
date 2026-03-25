#pragma once

#include "../Headers/nn.h"
#include "patches.h"

nn::Matrix BuildPatchTrainingData(const PatchSet& patch_set);
PatchList ExtractLatentCodes(nn::NeuralNetwork& autoencoder,
                             const PatchSet& patch_set,
                             nn::Activation hidden_act,
                             nn::Activation output_act);
PatchList DecodeLatentCodes(nn::NeuralNetwork& decoder,
                            const PatchList& latent_codes,
                            size_t patch_dim,
                            nn::Activation hidden_act,
                            nn::Activation output_act);
nn::NeuralNetwork ExtractDecoder(const nn::NeuralNetwork& autoencoder);
