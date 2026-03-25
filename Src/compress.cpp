#include "compress.h"

#include <iostream>

#include "../Headers/nn.h"
#include "codec.h"
#include "image_io.h"
#include "ninc_format.h"
#include "patches.h"

int RunCompress(const std::string& input_path,
                const std::string& model_path,
                const std::string& output_path) {
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
  const std::vector<size_t> arch = {patch_dim, 128, 32, 128, patch_dim};
  nn::NeuralNetwork autoencoder(arch);
  autoencoder.randomize(-0.5f, 0.5f);

  nn::Batch batch;
  const size_t epochs = 500;
  const float learning_rate = 0.005f;
  constexpr nn::Activation hidden_act = nn::Activation::Tanh;
  constexpr nn::Activation output_act = nn::Activation::Sigmoid;

  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    batch.process(train_data.rows, autoencoder, train_data, learning_rate, hidden_act, output_act);
    if (epoch % 25 == 0 || epoch + 1 == epochs) {
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

  nn::NeuralNetwork preview_decoder = decoder;
  const PatchList decoded_patches =
      DecodeLatentCodes(preview_decoder, latent_codes, patch_dim, hidden_act, output_act);
  const Image output = ReconstructFromPatches(patch_set, decoded_patches);
  if (!SavePNG(output_path, output)) {
    std::cerr << "Error: Could not save " << output_path << "!" << std::endl;
    return 1;
  }

  std::cout << "Saved compressed model to " << model_path << "!\n";
  std::cout << "Patch autoencoder reconstruction saved to " << output_path << "!\n";
  return 0;
}
