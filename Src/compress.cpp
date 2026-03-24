#include "compress.h"

#include <cstddef>
#include <iostream>
#include <vector>

#include "../Headers/nn.h"
#include "decompress.h"
#include "image_io.h"

int RunCompress(const std::string& input_path, const std::string& model_path,
                const std::string& output_path) {
  Image image;
  if (!LoadImage(input_path, image, 3)) {
    std::cerr << "Error: Could not load " << input_path << "!" << std::endl;
    return 1;
  }

  const int width = image.width;
  const int height = image.height;

  std::cout << "Image loaded successfully!\n";
  std::cout << "Width: " << width << "px\n";
  std::cout << "Height: " << height << "px\n";
  std::cout << "Total Pixels: " << (width * height) << "\n";

  nn::Matrix train_data(width * height, 5, 0.0f);
  for (size_t j = 0; j < static_cast<size_t>(height); ++j) {
    for (size_t i = 0; i < static_cast<size_t>(width); ++i) {
      const size_t row_idx = j * static_cast<size_t>(width) + i;

      train_data(row_idx, 0) = static_cast<float>(i) / (width - 1);
      train_data(row_idx, 1) = static_cast<float>(j) / (height - 1);

      const size_t img_idx = row_idx * 3;
      train_data(row_idx, 2) = image.pixels[img_idx + 0] / 255.0f;
      train_data(row_idx, 3) = image.pixels[img_idx + 1] / 255.0f;
      train_data(row_idx, 4) = image.pixels[img_idx + 2] / 255.0f;
    }
  }

  std::vector<size_t> arch = {2, 64, 64, 3};
  nn::NeuralNetwork compressor(arch);
  compressor.randomize(-0.5f, 0.5f);

  nn::Batch batch;
  const size_t epochs = 500;
  const float learning_rate = 0.01f;

  std::cout << "Starting compression...\n";

  for (size_t i = 0; i < epochs; ++i) {
    batch.process(train_data.rows, compressor, train_data, learning_rate,
                  nn::Activation::Relu, nn::Activation::Sigmoid);
    if (i % 100 == 0) {
      std::cout << "Epoch: " << i << " | Cost: " << batch.cost << "\n";
    }
  }

  std::cout << "Compression complete!\n";

  if (!compressor.save(model_path)) {
    std::cerr << "Error: Could not save " << model_path << "!" << std::endl;
    return 1;
  }

  const Image output = ReconstructImage(compressor, width, height);
  if (!SavePNG(output_path, output)) {
    std::cerr << "Error: Could not save " << output_path << "!" << std::endl;
    return 1;
  }

  std::cout << "Neural reconstruction saved to " << output_path << "!\n";
  return 0;
}
