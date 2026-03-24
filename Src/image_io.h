#pragma once

#include <string>
#include <vector>

struct Image {
  int width = 0;
  int height = 0;
  int channels = 0;
  std::vector<unsigned char> pixels;
};

bool LoadImage(const std::string& path, Image& image, int desired_channels = 3);
bool SavePNG(const std::string& path, const Image& image);
