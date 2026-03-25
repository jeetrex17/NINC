#include "ninc_format.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

constexpr std::array<char, 4> kNincMagic = {'N', 'I', 'N', 'C'};
constexpr std::uint32_t kNincVersion = 1;

template <typename T> bool WriteScalar(std::ofstream& out, const T& value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
  return out.good();
}

template <typename T> bool ReadScalar(std::ifstream& in, T& value) {
  return static_cast<bool>(in.read(reinterpret_cast<char*>(&value), sizeof(value)));
}

bool WriteSizeVector(std::ofstream& out, const std::vector<size_t>& values) {
  const std::uint64_t count = values.size();
  if (!WriteScalar(out, count)) {
    return false;
  }

  for (size_t value : values) {
    const std::uint64_t raw = value;
    if (!WriteScalar(out, raw)) {
      return false;
    }
  }

  return true;
}

bool ReadSizeVector(std::ifstream& in, std::vector<size_t>& values) {
  std::uint64_t count = 0;
  if (!ReadScalar(in, count)) {
    return false;
  }

  values.clear();
  values.reserve(static_cast<size_t>(count));
  for (std::uint64_t i = 0; i < count; ++i) {
    std::uint64_t raw = 0;
    if (!ReadScalar(in, raw)) {
      return false;
    }
    values.push_back(static_cast<size_t>(raw));
  }

  return true;
}

float DequantizeByte(std::uint8_t quantized, float min_value, float max_value) {
  if (max_value <= min_value) {
    return min_value;
  }

  const float normalized = static_cast<float>(quantized) / 255.0f;
  return min_value + normalized * (max_value - min_value);
}

std::uint8_t QuantizeToByte(float value, float min_value, float max_value) {
  if (max_value <= min_value) {
    return 0;
  }

  const float normalized = std::clamp((value - min_value) / (max_value - min_value), 0.0f, 1.0f);
  return static_cast<std::uint8_t>(std::lround(normalized * 255.0f));
}

bool WriteFloatVector(std::ofstream& out, const std::vector<float>& values) {
  const std::uint64_t count = values.size();
  if (!WriteScalar(out, count)) {
    return false;
  }

  out.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
  return out.good();
}

bool ReadFloatVector(std::ifstream& in, std::vector<float>& values) {
  std::uint64_t count = 0;
  if (!ReadScalar(in, count)) {
    return false;
  }

  values.resize(static_cast<size_t>(count));
  return static_cast<bool>(in.read(reinterpret_cast<char*>(values.data()),
                                   static_cast<std::streamsize>(values.size() * sizeof(float))));
}

bool WriteQuantizedMatrix(std::ofstream& out, const nn::Matrix& matrix) {
  const std::uint64_t rows = matrix.rows;
  const std::uint64_t cols = matrix.cols;
  if (!WriteScalar(out, rows) || !WriteScalar(out, cols)) {
    return false;
  }

  const auto [min_it, max_it] = std::minmax_element(matrix.data.begin(), matrix.data.end());
  const float min_value = matrix.data.empty() ? 0.0f : *min_it;
  const float max_value = matrix.data.empty() ? 0.0f : *max_it;
  if (!WriteScalar(out, min_value) || !WriteScalar(out, max_value)) {
    return false;
  }

  std::vector<std::uint8_t> quantized(matrix.data.size(), 0);
  for (size_t i = 0; i < matrix.data.size(); ++i) {
    quantized[i] = QuantizeToByte(matrix.data[i], min_value, max_value);
  }

  out.write(reinterpret_cast<const char*>(quantized.data()),
            static_cast<std::streamsize>(quantized.size()));
  return out.good();
}

bool ReadQuantizedMatrix(std::ifstream& in, nn::Matrix& matrix) {
  std::uint64_t rows = 0;
  std::uint64_t cols = 0;
  float min_value = 0.0f;
  float max_value = 0.0f;
  if (!ReadScalar(in, rows) || !ReadScalar(in, cols) || !ReadScalar(in, min_value) ||
      !ReadScalar(in, max_value)) {
    return false;
  }

  if (rows != matrix.rows || cols != matrix.cols) {
    return false;
  }

  std::vector<std::uint8_t> quantized(matrix.data.size(), 0);
  if (!in.read(reinterpret_cast<char*>(quantized.data()),
               static_cast<std::streamsize>(quantized.size()))) {
    return false;
  }

  for (size_t i = 0; i < matrix.data.size(); ++i) {
    matrix.data[i] = DequantizeByte(quantized[i], min_value, max_value);
  }

  return true;
}

bool WriteQuantizedLatents(std::ofstream& out,
                           const PatchList& latent_codes,
                           std::uint64_t latent_size) {
  std::vector<float> min_values(static_cast<size_t>(latent_size),
                                std::numeric_limits<float>::infinity());
  std::vector<float> max_values(static_cast<size_t>(latent_size),
                                -std::numeric_limits<float>::infinity());

  for (const auto& latent_code : latent_codes) {
    if (latent_code.size() != latent_size) {
      return false;
    }

    for (size_t i = 0; i < latent_code.size(); ++i) {
      min_values[i] = std::min(min_values[i], latent_code[i]);
      max_values[i] = std::max(max_values[i], latent_code[i]);
    }
  }

  for (size_t i = 0; i < min_values.size(); ++i) {
    if (!std::isfinite(min_values[i]) || !std::isfinite(max_values[i])) {
      min_values[i] = 0.0f;
      max_values[i] = 0.0f;
    }
  }

  if (!WriteFloatVector(out, min_values) || !WriteFloatVector(out, max_values)) {
    return false;
  }

  std::vector<std::uint8_t> quantized_row(static_cast<size_t>(latent_size), 0);
  for (const auto& latent_code : latent_codes) {
    for (size_t i = 0; i < latent_code.size(); ++i) {
      quantized_row[i] = QuantizeToByte(latent_code[i], min_values[i], max_values[i]);
    }

    out.write(reinterpret_cast<const char*>(quantized_row.data()),
              static_cast<std::streamsize>(quantized_row.size()));
    if (!out.good()) {
      return false;
    }
  }

  return true;
}

bool ReadQuantizedLatents(std::ifstream& in,
                          std::uint64_t patch_count,
                          std::uint64_t latent_size,
                          PatchList& latent_codes) {
  std::vector<float> min_values;
  std::vector<float> max_values;
  if (!ReadFloatVector(in, min_values) || !ReadFloatVector(in, max_values)) {
    return false;
  }

  if (min_values.size() != latent_size || max_values.size() != latent_size) {
    return false;
  }

  latent_codes.assign(static_cast<size_t>(patch_count),
                      Patch(static_cast<size_t>(latent_size), 0.0f));
  std::vector<std::uint8_t> quantized_row(static_cast<size_t>(latent_size), 0);
  for (auto& latent_code : latent_codes) {
    if (!in.read(reinterpret_cast<char*>(quantized_row.data()),
                 static_cast<std::streamsize>(quantized_row.size()))) {
      return false;
    }

    for (size_t i = 0; i < latent_code.size(); ++i) {
      latent_code[i] = DequantizeByte(quantized_row[i], min_values[i], max_values[i]);
    }
  }

  return true;
}

bool IsValidActivation(std::uint32_t value) {
  using ActivationInt = std::underlying_type_t<nn::Activation>;
  const auto max_value =
      static_cast<std::uint32_t>(static_cast<ActivationInt>(nn::Activation::Sin));
  return value <= max_value;
}

} // namespace

bool SaveNinc(const std::string& path,
              const PatchSet& patch_set,
              const nn::NeuralNetwork& decoder,
              const PatchList& latent_codes,
              nn::Activation hidden_act,
              nn::Activation output_act) {
  if (decoder.arch.empty()) {
    return false;
  }

  if (latent_codes.size() != patch_set.xs.size()) {
    return false;
  }

  const std::uint64_t patch_count = latent_codes.size();
  const std::uint64_t latent_size = latent_codes.empty() ? 0 : latent_codes.front().size();

  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    return false;
  }

  out.write(kNincMagic.data(), static_cast<std::streamsize>(kNincMagic.size()));
  if (!out.good() || !WriteScalar(out, kNincVersion)) {
    return false;
  }

  if (!WriteScalar(out, static_cast<std::uint32_t>(patch_set.original_width)) ||
      !WriteScalar(out, static_cast<std::uint32_t>(patch_set.original_height)) ||
      !WriteScalar(out, static_cast<std::uint32_t>(patch_set.padded_width)) ||
      !WriteScalar(out, static_cast<std::uint32_t>(patch_set.padded_height)) ||
      !WriteScalar(out, static_cast<std::uint32_t>(patch_set.patch_size)) ||
      !WriteScalar(out, static_cast<std::uint32_t>(patch_set.stride)) ||
      !WriteScalar(out, static_cast<std::uint32_t>(patch_set.channels)) ||
      !WriteScalar(out,
                   static_cast<std::uint32_t>(
                       static_cast<std::underlying_type_t<nn::Activation>>(hidden_act))) ||
      !WriteScalar(out,
                   static_cast<std::uint32_t>(
                       static_cast<std::underlying_type_t<nn::Activation>>(output_act))) ||
      !WriteScalar(out, patch_count) || !WriteScalar(out, latent_size)) {
    return false;
  }

  if (!WriteSizeVector(out, decoder.arch)) {
    return false;
  }

  for (const auto& w : decoder.ws) {
    if (!WriteQuantizedMatrix(out, w)) {
      return false;
    }
  }

  for (const auto& b : decoder.bs) {
    if (!WriteQuantizedMatrix(out, b)) {
      return false;
    }
  }

  if (!WriteQuantizedLatents(out, latent_codes, latent_size)) {
    return false;
  }

  return out.good();
}

bool LoadNinc(const std::string& path, NincData& data) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    return false;
  }

  std::array<char, 4> magic{};
  if (!in.read(magic.data(), static_cast<std::streamsize>(magic.size())) || magic != kNincMagic) {
    return false;
  }

  std::uint32_t version = 0;
  if (!ReadScalar(in, version) || version != kNincVersion) {
    return false;
  }

  std::uint32_t original_width = 0;
  std::uint32_t original_height = 0;
  std::uint32_t padded_width = 0;
  std::uint32_t padded_height = 0;
  std::uint32_t patch_size = 0;
  std::uint32_t stride = 0;
  std::uint32_t channels = 0;
  std::uint32_t hidden_act = 0;
  std::uint32_t output_act = 0;
  std::uint64_t patch_count = 0;
  std::uint64_t latent_size = 0;

  if (!ReadScalar(in, original_width) || !ReadScalar(in, original_height) ||
      !ReadScalar(in, padded_width) || !ReadScalar(in, padded_height) ||
      !ReadScalar(in, patch_size) || !ReadScalar(in, stride) || !ReadScalar(in, channels) ||
      !ReadScalar(in, hidden_act) || !ReadScalar(in, output_act) || !ReadScalar(in, patch_count) ||
      !ReadScalar(in, latent_size)) {
    return false;
  }

  if (!IsValidActivation(hidden_act) || !IsValidActivation(output_act)) {
    return false;
  }

  std::vector<size_t> decoder_arch;
  if (!ReadSizeVector(in, decoder_arch) || decoder_arch.size() < 2) {
    return false;
  }

  std::vector<nn::Matrix> decoder_ws;
  std::vector<nn::Matrix> decoder_bs;
  for (size_t i = 1; i < decoder_arch.size(); ++i) {
    decoder_ws.emplace_back(decoder_arch[i - 1], decoder_arch[i]);
    decoder_bs.emplace_back(1, decoder_arch[i]);
  }

  for (auto& w : decoder_ws) {
    if (!ReadQuantizedMatrix(in, w)) {
      return false;
    }
  }

  for (auto& b : decoder_bs) {
    if (!ReadQuantizedMatrix(in, b)) {
      return false;
    }
  }

  PatchList latent_codes;
  if (!ReadQuantizedLatents(in, patch_count, latent_size, latent_codes)) {
    return false;
  }

  data.original_width = static_cast<int>(original_width);
  data.original_height = static_cast<int>(original_height);
  data.padded_width = static_cast<int>(padded_width);
  data.padded_height = static_cast<int>(padded_height);
  data.patch_size = static_cast<int>(patch_size);
  data.stride = static_cast<int>(stride);
  data.channels = static_cast<int>(channels);
  data.hidden_act = static_cast<nn::Activation>(hidden_act);
  data.output_act = static_cast<nn::Activation>(output_act);
  data.decoder_arch = std::move(decoder_arch);
  data.decoder_ws = std::move(decoder_ws);
  data.decoder_bs = std::move(decoder_bs);
  data.latent_codes = std::move(latent_codes);
  return true;
}

nn::NeuralNetwork BuildDecoderNetwork(const NincData& data) {
  assert(data.decoder_arch.size() >= 2);
  assert(data.decoder_ws.size() + 1 == data.decoder_arch.size());
  assert(data.decoder_bs.size() + 1 == data.decoder_arch.size());

  nn::NeuralNetwork decoder(data.decoder_arch);
  decoder.ws = data.decoder_ws;
  decoder.bs = data.decoder_bs;
  return decoder;
}
