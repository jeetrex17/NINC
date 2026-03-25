#include "ninc_format.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <type_traits>

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

bool WriteMatrix(std::ofstream& out, const nn::Matrix& matrix) {
  const std::uint64_t rows = matrix.rows;
  const std::uint64_t cols = matrix.cols;
  if (!WriteScalar(out, rows) || !WriteScalar(out, cols)) {
    return false;
  }

  out.write(reinterpret_cast<const char*>(matrix.data.data()), matrix.data.size() * sizeof(float));
  return out.good();
}

bool ReadMatrix(std::ifstream& in, nn::Matrix& matrix) {
  std::uint64_t rows = 0;
  std::uint64_t cols = 0;
  if (!ReadScalar(in, rows) || !ReadScalar(in, cols)) {
    return false;
  }

  if (rows != matrix.rows || cols != matrix.cols) {
    return false;
  }

  return static_cast<bool>(
      in.read(reinterpret_cast<char*>(matrix.data.data()), matrix.data.size() * sizeof(float)));
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
    if (!WriteMatrix(out, w)) {
      return false;
    }
  }

  for (const auto& b : decoder.bs) {
    if (!WriteMatrix(out, b)) {
      return false;
    }
  }

  for (const auto& latent_code : latent_codes) {
    if (latent_code.size() != latent_size) {
      return false;
    }

    out.write(reinterpret_cast<const char*>(latent_code.data()),
              static_cast<std::streamsize>(latent_code.size() * sizeof(float)));
    if (!out.good()) {
      return false;
    }
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
    if (!ReadMatrix(in, w)) {
      return false;
    }
  }

  for (auto& b : decoder_bs) {
    if (!ReadMatrix(in, b)) {
      return false;
    }
  }

  PatchList latent_codes(static_cast<size_t>(patch_count),
                         Patch(static_cast<size_t>(latent_size), 0.0f));
  for (auto& latent_code : latent_codes) {
    if (!in.read(reinterpret_cast<char*>(latent_code.data()),
                 static_cast<std::streamsize>(latent_code.size() * sizeof(float)))) {
      return false;
    }
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
