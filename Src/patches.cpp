#include "patches.h"

#include <algorithm>
#include <cassert>
#include <cstddef>

namespace {

int ComputePaddedDim(int original_dim, int patch_size, int stride) {
  int padded_dim = std::max(original_dim, patch_size);
  while ((padded_dim - patch_size) % stride != 0) {
    ++padded_dim;
  }
  return padded_dim;
}

Image CreatePaddedImage(const Image& input, int padded_width,
                        int padded_height) {
  Image padded;
  padded.width = padded_width;
  padded.height = padded_height;
  padded.channels = input.channels;
  padded.pixels.assign(
      static_cast<size_t>(padded_width) * padded_height * padded.channels, 0);

  for (int y = 0; y < input.height; ++y) {
    for (int x = 0; x < input.width; ++x) {
      for (int c = 0; c < input.channels; ++c) {
        const size_t src_idx =
            (static_cast<size_t>(y) * input.width + x) * input.channels + c;
        const size_t dst_idx =
            (static_cast<size_t>(y) * padded_width + x) * padded.channels + c;
        padded.pixels[dst_idx] = input.pixels[src_idx];
      }
    }
  }

  return padded;
}

size_t FlatPatchIndex(int px, int py, int channel, int patch_size,
                      int channels) {
  return (static_cast<size_t>(py) * patch_size + px) * channels + channel;
}

}  // namespace

PatchSet ExtractPatches(const Image& image, int patch_size, int stride) {
  assert(image.channels == 3);

  PatchSet patch_set;
  patch_set.original_width = image.width;
  patch_set.original_height = image.height;
  patch_set.patch_size = patch_size;
  patch_set.stride = stride;
  patch_set.channels = image.channels;
  patch_set.padded_width = ComputePaddedDim(image.width, patch_size, stride);
  patch_set.padded_height = ComputePaddedDim(image.height, patch_size, stride);

  const Image padded =
      CreatePaddedImage(image, patch_set.padded_width, patch_set.padded_height);

  for (int y = 0; y <= patch_set.padded_height - patch_size; y += stride) {
    for (int x = 0; x <= patch_set.padded_width - patch_size; x += stride) {
      std::vector<float> patch(
          static_cast<size_t>(patch_size) * patch_size * image.channels, 0.0f);

      for (int py = 0; py < patch_size; ++py) {
        for (int px = 0; px < patch_size; ++px) {
          for (int c = 0; c < image.channels; ++c) {
            const size_t src_idx =
                (static_cast<size_t>(y + py) * padded.width + (x + px)) *
                    padded.channels +
                c;
            const size_t dst_idx =
                FlatPatchIndex(px, py, c, patch_size, image.channels);
            patch[dst_idx] = padded.pixels[src_idx] / 255.0f;
          }
        }
      }

      patch_set.xs.push_back(x);
      patch_set.ys.push_back(y);
      patch_set.patches.push_back(std::move(patch));
    }
  }

  return patch_set;
}

Image ReconstructFromPatches(
    const PatchSet& patch_set,
    const std::vector<std::vector<float>>& decoded_patches) {
  assert(decoded_patches.size() == patch_set.patches.size());

  const size_t padded_pixel_count =
      static_cast<size_t>(patch_set.padded_width) * patch_set.padded_height;
  std::vector<float> accum(padded_pixel_count * patch_set.channels, 0.0f);
  std::vector<float> counts(padded_pixel_count, 0.0f);

  for (size_t patch_idx = 0; patch_idx < decoded_patches.size(); ++patch_idx) {
    const auto& patch = decoded_patches[patch_idx];
    assert(
        patch.size() == static_cast<size_t>(patch_set.patch_size) *
                            patch_set.patch_size * patch_set.channels);

    const int start_x = patch_set.xs[patch_idx];
    const int start_y = patch_set.ys[patch_idx];

    for (int py = 0; py < patch_set.patch_size; ++py) {
      for (int px = 0; px < patch_set.patch_size; ++px) {
        const size_t pixel_idx =
            static_cast<size_t>(start_y + py) * patch_set.padded_width +
            (start_x + px);
        counts[pixel_idx] += 1.0f;

        for (int c = 0; c < patch_set.channels; ++c) {
          const size_t src_idx = FlatPatchIndex(
              px, py, c, patch_set.patch_size, patch_set.channels);
          const size_t dst_idx = pixel_idx * patch_set.channels + c;
          accum[dst_idx] += std::clamp(patch[src_idx], 0.0f, 1.0f);
        }
      }
    }
  }

  Image output;
  output.width = patch_set.original_width;
  output.height = patch_set.original_height;
  output.channels = patch_set.channels;
  output.pixels.resize(
      static_cast<size_t>(output.width) * output.height * output.channels);

  for (int y = 0; y < output.height; ++y) {
    for (int x = 0; x < output.width; ++x) {
      const size_t padded_pixel_idx =
          static_cast<size_t>(y) * patch_set.padded_width + x;
      const float count = std::max(counts[padded_pixel_idx], 1.0f);

      for (int c = 0; c < output.channels; ++c) {
        const size_t src_idx = padded_pixel_idx * output.channels + c;
        const size_t dst_idx =
            (static_cast<size_t>(y) * output.width + x) * output.channels + c;
        const float avg = accum[src_idx] / count;
        output.pixels[dst_idx] = static_cast<unsigned char>(
            std::clamp(avg, 0.0f, 1.0f) * 255.0f + 0.5f);
      }
    }
  }

  return output;
}
