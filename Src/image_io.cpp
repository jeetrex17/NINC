#include "image_io.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../Headers/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../Headers/stb_image_write.h"

bool LoadImage(const std::string& path, Image& image, int desired_channels) {
  int width = 0;
  int height = 0;
  int channels = 0;
  unsigned char* img_data = stbi_load(path.c_str(), &width, &height, &channels, desired_channels);
  if (img_data == nullptr) {
    return false;
  }

  image.width = width;
  image.height = height;
  image.channels = desired_channels;
  image.pixels.assign(img_data, img_data + static_cast<size_t>(width) * height * desired_channels);

  stbi_image_free(img_data);
  return true;
}

bool SavePNG(const std::string& path, const Image& image) {
  return stbi_write_png(path.c_str(),
                        image.width,
                        image.height,
                        image.channels,
                        image.pixels.data(),
                        image.width * image.channels) != 0;
}
