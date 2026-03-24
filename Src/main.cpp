#include <cstddef>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "../Headers/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../Headers/stb_image_write.h"

#include "../Headers/nn.h"

int main() {
  int width, height, channels;

  unsigned char *img_data =
      stbi_load("test.png", &width, &height, &channels, 3);

  if (img_data == nullptr) {
    std::cerr << "Error: Could not load test.png!" << std::endl;
    return 1;
  }

  std::cout << "Image loaded successfully!\n";
  std::cout << "Width: " << width << "px\n";
  std::cout << "Height: " << height << "px\n";
  std::cout << "Total Pixels: " << (width * height) << "\n";

  /*
  Column 0: The X coordinate (Input 1)
  Column 1: The Y coordinate (Input 2)
  Column 2: The Red color (Target 1)
  Column 3: The Green color (Target 2)
  Column 4: The Blue color (Target 3)
  There will be width*height Rows so if image is 64X64 so 4096 Rows;
  */
  nn::Matrix train_data(width * height, 5, 0.0f);
  for (size_t j = 0; j < height; j++) {
    for (size_t i = 0; i < width; i++) {
      size_t row_idx = j * width + i;

      // normalising coordinates
      train_data(row_idx, 0) = (float)i / (width - 1);
      train_data(row_idx, 1) = (float)j / (height - 1);

      // Memory looks like: [R, G, B,  R, G, B,  R, G, B...]
      // So we multiply by 3 to find this exact pixel's starting index
      size_t img_idx = row_idx * 3;

      // normalising colors data
      train_data(row_idx, 2) = img_data[img_idx + 0] / 255.0f;
      train_data(row_idx, 3) = img_data[img_idx + 1] / 255.0f;
      train_data(row_idx, 4) = img_data[img_idx + 2] / 255.0f;
    }
  }
  stbi_image_free(img_data);

  return 0;
}
