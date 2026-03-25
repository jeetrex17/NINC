#include "codec.h"

#include <cassert>

nn::Matrix BuildPatchTrainingData(const PatchSet& patch_set) {
  const size_t patch_dim =
      static_cast<size_t>(patch_set.patch_size) * patch_set.patch_size * patch_set.channels;
  nn::Matrix train_data(patch_set.patches.size(), patch_dim * 2, 0.0f);

  for (size_t patch_idx = 0; patch_idx < patch_set.patches.size(); ++patch_idx) {
    const auto& patch = patch_set.patches[patch_idx];
    for (size_t value_idx = 0; value_idx < patch_dim; ++value_idx) {
      train_data(patch_idx, value_idx) = patch[value_idx];
      train_data(patch_idx, patch_dim + value_idx) = patch[value_idx];
    }
  }

  return train_data;
}

PatchList ExtractLatentCodes(nn::NeuralNetwork& autoencoder,
                             const PatchSet& patch_set,
                             nn::Activation hidden_act,
                             nn::Activation output_act) {
  assert(!autoencoder.as.empty());

  const size_t latent_layer_idx = autoencoder.as.size() / 2;
  const size_t patch_dim =
      static_cast<size_t>(patch_set.patch_size) * patch_set.patch_size * patch_set.channels;
  const size_t latent_dim = autoencoder.as[latent_layer_idx].cols;
  PatchList latent_codes(patch_set.patches.size(), Patch(latent_dim, 0.0f));

  for (size_t patch_idx = 0; patch_idx < patch_set.patches.size(); ++patch_idx) {
    const auto& patch = patch_set.patches[patch_idx];

    for (size_t value_idx = 0; value_idx < patch_dim; ++value_idx) {
      autoencoder.get_input()(0, value_idx) = patch[value_idx];
    }

    autoencoder.forward(hidden_act, output_act);

    const nn::Matrix& latent = autoencoder.as[latent_layer_idx];
    for (size_t value_idx = 0; value_idx < latent_dim; ++value_idx) {
      latent_codes[patch_idx][value_idx] = latent(0, value_idx);
    }
  }

  return latent_codes;
}

PatchList DecodeLatentCodes(nn::NeuralNetwork& decoder,
                            const PatchList& latent_codes,
                            size_t patch_dim,
                            nn::Activation hidden_act,
                            nn::Activation output_act) {
  const size_t latent_dim = decoder.get_input().cols;
  PatchList decoded_patches(latent_codes.size(), Patch(patch_dim, 0.0f));

  for (size_t patch_idx = 0; patch_idx < latent_codes.size(); ++patch_idx) {
    const auto& latent_code = latent_codes[patch_idx];
    assert(latent_code.size() == latent_dim);

    for (size_t value_idx = 0; value_idx < latent_dim; ++value_idx) {
      decoder.get_input()(0, value_idx) = latent_code[value_idx];
    }

    decoder.forward(hidden_act, output_act);

    for (size_t value_idx = 0; value_idx < patch_dim; ++value_idx) {
      decoded_patches[patch_idx][value_idx] = decoder.get_output()(0, value_idx);
    }
  }

  return decoded_patches;
}

nn::NeuralNetwork ExtractDecoder(const nn::NeuralNetwork& autoencoder) {
  assert(autoencoder.arch.size() >= 3);

  const size_t latent_layer_idx = autoencoder.arch.size() / 2;
  const std::vector<size_t> decoder_arch(autoencoder.arch.begin() + latent_layer_idx,
                                         autoencoder.arch.end());
  nn::NeuralNetwork decoder(decoder_arch);

  for (size_t layer_idx = latent_layer_idx; layer_idx < autoencoder.ws.size(); ++layer_idx) {
    decoder.ws[layer_idx - latent_layer_idx] = autoencoder.ws[layer_idx];
    decoder.bs[layer_idx - latent_layer_idx] = autoencoder.bs[layer_idx];
  }

  return decoder;
}
