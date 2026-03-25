#include "compress.h"

#include <filesystem>
#include <iostream>

#include "../Headers/nn.h"
#include "codec.h"
#include "image_io.h"
#include "ninc_format.h"
#include "patches.h"

int RunCompress(const std::string& input_path, const std::string& model_path) {
  Image image;
  if (!LoadImage(input_path, image, 3)) {
    std::cerr << "Error: Could not load " << input_path << "!" << std::endl;
    return 1;
  }

  std::cout << "Image loaded successfully!\n";
  std::cout << "Width: " << image.width << "px\n";
  std::cout << "Height: " << image.height << "px\n";
  std::cout << "Total Pixels: " << (image.width * image.height) << "\n";

  const PatchSet patch_set = ExtractPatches(image, 8, 8);
  const size_t patch_dim =
      static_cast<size_t>(patch_set.patch_size) * patch_set.patch_size * patch_set.channels;
  std::cout << "Patch autoencoder training...\n";
  std::cout << "Padded Size: " << patch_set.padded_width << "x" << patch_set.padded_height << "\n";
  std::cout << "Patch Size: " << patch_set.patch_size << " | Stride: " << patch_set.stride << "\n";
  std::cout << "Patch Count: " << patch_set.patches.size() << "\n";
  std::cout << "Patch Dim: " << patch_dim << "\n";

  const nn::Matrix train_data = BuildPatchTrainingData(patch_set);
  const std::vector<size_t> arch = {patch_dim, 128, 32, 128, patch_dim};
  nn::NeuralNetwork autoencoder(arch);
  autoencoder.randomize(-0.5f, 0.5f);

  nn::Batch batch;
  const size_t epochs = 200;
  const float learning_rate = 0.005f;
  constexpr nn::Activation hidden_act = nn::Activation::Tanh;
  constexpr nn::Activation output_act = nn::Activation::Sigmoid;
  constexpr size_t log_interval = 25;

  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    const bool should_log = (epoch % log_interval == 0) || (epoch + 1 == epochs);
    batch.process(train_data.rows,
                  autoencoder,
                  train_data,
                  learning_rate,
                  should_log,
                  hidden_act,
                  output_act);
    if (should_log) {
      std::cout << "Epoch: " << epoch << " | Cost: " << batch.cost << std::endl;
    }
  }

  const PatchList latent_codes = ExtractLatentCodes(autoencoder, patch_set, hidden_act, output_act);
  std::cout << "Latent code count: " << latent_codes.size() << "\n";
  std::cout << "Latent size: " << (latent_codes.empty() ? 0 : latent_codes.front().size()) << "\n";

  const nn::NeuralNetwork decoder = ExtractDecoder(autoencoder);
  if (!SaveNinc(model_path, patch_set, decoder, latent_codes, hidden_act, output_act)) {
    std::cerr << "Error: Could not save " << model_path << "!" << std::endl;
    return 1;
  }

  const size_t raw_size = static_cast<size_t>(image.width) * image.height * image.channels;
  const auto ninc_size = std::filesystem::file_size(model_path);
  std::cout << "\n--- Compression Stats ---\n";
  std::cout << "Raw image: " << raw_size << " bytes\n";
  std::cout << "Compressed: " << ninc_size << " bytes\n";
  std::cout << "Ratio: " << static_cast<float>(raw_size) / ninc_size << "x\n";
  std::cout << "Saved to " << model_path << "\n";

  return 0;
}
