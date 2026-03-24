#include "decompress.h"

#include <cstddef>

Image ReconstructImage(nn::NeuralNetwork& compressor, int width, int height) {
  Image output;
  output.width = width;
  output.height = height;
  output.channels = 3;
  output.pixels.resize(static_cast<size_t>(width) * height * output.channels);

  for (size_t j = 0; j < static_cast<size_t>(height); ++j) {
    for (size_t i = 0; i < static_cast<size_t>(width); ++i) {
      compressor.get_input()(0, 0) = static_cast<float>(i) / (width - 1);
      compressor.get_input()(0, 1) = static_cast<float>(j) / (height - 1);

      compressor.forward(nn::Activation::Relu, nn::Activation::Sigmoid);

      const size_t img_idx = (j * static_cast<size_t>(width) + i) * 3;
      output.pixels[img_idx + 0] =
          static_cast<unsigned char>(compressor.get_output()(0, 0) * 255.0f);
      output.pixels[img_idx + 1] =
          static_cast<unsigned char>(compressor.get_output()(0, 1) * 255.0f);
      output.pixels[img_idx + 2] =
          static_cast<unsigned char>(compressor.get_output()(0, 2) * 255.0f);
    }
  }

  return output;
}
