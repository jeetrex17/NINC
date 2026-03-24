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

  stbi_image_free(img_data);

  return 0;
}
