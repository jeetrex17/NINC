#pragma once

#include <vector>

#include "image_io.h"

using Patch = std::vector<float>;
using PatchList = std::vector<Patch>;

struct PatchSet {
  int original_width = 0;
  int original_height = 0;
  int padded_width = 0;
  int padded_height = 0;
  int patch_size = 8;
  int stride = 4;
  int channels = 3;

  PatchList patches;
  std::vector<int> xs;
  std::vector<int> ys;
};

PatchSet BuildPatchLayout(
    int original_width, int original_height, int patch_size = 8, int stride = 4, int channels = 3);
PatchSet ExtractPatches(const Image& image, int patch_size = 8, int stride = 4);
Image ReconstructFromPatches(const PatchSet& patch_set, const PatchList& decoded_patches);
