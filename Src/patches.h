#pragma once

#include <vector>

#include "image_io.h"

struct PatchSet {
  int original_width = 0;
  int original_height = 0;
  int padded_width = 0;
  int padded_height = 0;
  int patch_size = 8;
  int stride = 4;
  int channels = 3;

  std::vector<std::vector<float>> patches;
  std::vector<int> xs;
  std::vector<int> ys;
};

PatchSet ExtractPatches(const Image& image, int patch_size = 8, int stride = 4);
Image ReconstructFromPatches(const PatchSet& patch_set,
                             const std::vector<std::vector<float>>& decoded_patches);
