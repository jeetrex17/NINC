#include "compress.h"

#include <cassert>
#include <iostream>

#include "../Headers/nn.h"
#include "image_io.h"
#include "patches.h"

namespace {

using Patch = std::vector<float>;
using PatchList = std::vector<Patch>;

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

PatchList DecodeLatentCodes(nn::NeuralNetwork& autoencoder,
                            const PatchList& latent_codes,
                            size_t patch_dim,
                            nn::Activation hidden_act,
                            nn::Activation output_act) {
  assert(!autoencoder.as.empty());
  assert(autoencoder.ws.size() + 1 == autoencoder.as.size());

  const size_t latent_layer_idx = autoencoder.as.size() / 2;
  const size_t latent_dim = autoencoder.as[latent_layer_idx].cols;
  PatchList decoded_patches(latent_codes.size(), Patch(patch_dim, 0.0f));

  for (size_t patch_idx = 0; patch_idx < latent_codes.size(); ++patch_idx) {
    const auto& latent_code = latent_codes[patch_idx];
    assert(latent_code.size() == latent_dim);

    for (size_t value_idx = 0; value_idx < latent_dim; ++value_idx) {
      autoencoder.as[latent_layer_idx](0, value_idx) = latent_code[value_idx];
    }

    for (size_t layer_idx = latent_layer_idx; layer_idx < autoencoder.ws.size(); ++layer_idx) {
      autoencoder.as[layer_idx + 1] =
          nn::Matrix::dot_mt(autoencoder.as[layer_idx], autoencoder.ws[layer_idx]);
      autoencoder.as[layer_idx + 1] += autoencoder.bs[layer_idx];
      autoencoder.zs[layer_idx] = autoencoder.as[layer_idx + 1];

      const nn::Activation layer_act =
          (layer_idx == autoencoder.ws.size() - 1) ? output_act : hidden_act;
      autoencoder.as[layer_idx + 1].apply_activation(layer_act);
    }

    for (size_t value_idx = 0; value_idx < patch_dim; ++value_idx) {
      decoded_patches[patch_idx][value_idx] = autoencoder.get_output()(0, value_idx);
    }
  }

  return decoded_patches;
}

} // namespace

int RunCompress(const std::string& input_path,
                const std::string& model_path,
                const std::string& output_path) {
  (void)model_path;

  Image image;
  if (!LoadImage(input_path, image, 3)) {
    std::cerr << "Error: Could not load " << input_path << "!" << std::endl;
    return 1;
  }

  std::cout << "Image loaded successfully!\n";
  std::cout << "Width: " << image.width << "px\n";
  std::cout << "Height: " << image.height << "px\n";
  std::cout << "Total Pixels: " << (image.width * image.height) << "\n";

  const PatchSet patch_set = ExtractPatches(image, 8, 4);
  const size_t patch_dim =
      static_cast<size_t>(patch_set.patch_size) * patch_set.patch_size * patch_set.channels;
  std::cout << "Patch autoencoder training...\n";
  std::cout << "Padded Size: " << patch_set.padded_width << "x" << patch_set.padded_height << "\n";
  std::cout << "Patch Size: " << patch_set.patch_size << " | Stride: " << patch_set.stride << "\n";
  std::cout << "Patch Count: " << patch_set.patches.size() << "\n";
  std::cout << "Patch Dim: " << patch_dim << "\n";

  const nn::Matrix train_data = BuildPatchTrainingData(patch_set);
  const std::vector<size_t> arch = {patch_dim, 64, 16, 64, patch_dim};
  nn::NeuralNetwork autoencoder(arch);
  autoencoder.randomize(-0.5f, 0.5f);

  nn::Batch batch;
  // Keep the first patch-autoencoder pass cheap enough to iterate on.
  const size_t epochs = 200;
  const float learning_rate = 0.01f;
  constexpr nn::Activation hidden_act = nn::Activation::Tanh;
  constexpr nn::Activation output_act = nn::Activation::Sigmoid;

  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    batch.process(train_data.rows, autoencoder, train_data, learning_rate, hidden_act, output_act);
    if (epoch % 25 == 0 || epoch + 1 == epochs) {
      std::cout << "Epoch: " << epoch << " | Cost: " << batch.cost << std::endl;
    }
  }

  const auto latent_codes = ExtractLatentCodes(autoencoder, patch_set, hidden_act, output_act);
  std::cout << "Latent code count: " << latent_codes.size() << "\n";
  std::cout << "Latent size: " << (latent_codes.empty() ? 0 : latent_codes.front().size()) << "\n";

  const auto decoded_patches =
      DecodeLatentCodes(autoencoder, latent_codes, patch_dim, hidden_act, output_act);
  const Image output = ReconstructFromPatches(patch_set, decoded_patches);
  if (!SavePNG(output_path, output)) {
    std::cerr << "Error: Could not save " << output_path << "!" << std::endl;
    return 1;
  }

  std::cout << "Patch autoencoder reconstruction saved to " << output_path << "!\n";
  return 0;
}
