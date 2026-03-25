#pragma once

#include <string>
#include <vector>

#include "../Headers/nn.h"
#include "patches.h"

struct NincData {
  int original_width = 0;
  int original_height = 0;
  int padded_width = 0;
  int padded_height = 0;
  int patch_size = 8;
  int stride = 4;
  int channels = 3;
  nn::Activation hidden_act = nn::Activation::Tanh;
  nn::Activation output_act = nn::Activation::Sigmoid;
  std::vector<size_t> decoder_arch;
  std::vector<nn::Matrix> decoder_ws;
  std::vector<nn::Matrix> decoder_bs;
  PatchList latent_codes;
};

bool SaveNinc(const std::string& path,
              const PatchSet& patch_set,
              const nn::NeuralNetwork& decoder,
              const PatchList& latent_codes,
              nn::Activation hidden_act,
              nn::Activation output_act);
bool LoadNinc(const std::string& path, NincData& data);
nn::NeuralNetwork BuildDecoderNetwork(const NincData& data);
