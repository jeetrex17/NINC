#include "decompress.h"

#include <iostream>

#include "codec.h"
#include "image_io.h"
#include "ninc_format.h"
#include "patches.h"

int RunDecompress(const std::string& model_path, const std::string& output_path) {
  NincData data;
  if (!LoadNinc(model_path, data)) {
    std::cerr << "Error: Could not load " << model_path << "!" << std::endl;
    return 1;
  }

  PatchSet patch_set = BuildPatchLayout(
      data.original_width, data.original_height, data.patch_size, data.stride, data.channels);
  if (patch_set.padded_width != data.padded_width ||
      patch_set.padded_height != data.padded_height) {
    std::cerr << "Error: Saved patch layout does not match expected padded dimensions."
              << std::endl;
    return 1;
  }

  if (patch_set.xs.size() != data.latent_codes.size()) {
    std::cerr << "Error: Latent code count does not match patch layout." << std::endl;
    return 1;
  }

  nn::NeuralNetwork decoder = BuildDecoderNetwork(data);
  const size_t patch_dim =
      static_cast<size_t>(patch_set.patch_size) * patch_set.patch_size * patch_set.channels;
  const PatchList decoded_patches =
      DecodeLatentCodes(decoder, data.latent_codes, patch_dim, data.hidden_act, data.output_act);
  const Image output = ReconstructFromPatches(patch_set, decoded_patches);
  if (!SavePNG(output_path, output)) {
    std::cerr << "Error: Could not save " << output_path << "!" << std::endl;
    return 1;
  }

  std::cout << "Loaded compressed model from " << model_path << "!\n";
  std::cout << "Decompressed image saved to " << output_path << "!\n";
  return 0;
}
