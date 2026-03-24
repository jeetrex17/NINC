#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <ranges>
#include <thread>
#include <utility>
#include <vector>

#ifndef NN_RELU_PARAM
#define NN_RELU_PARAM 0.01f
#endif

namespace nn {

enum class Activation { Sigmoid, Relu, Tanh, Sin };
#ifndef NN_ACT
#define NN_ACT nn::Activation::Sigmoid
#endif

inline float Sigmoid(float x) { return (1 / (1 + exp(-x))); }
inline float Relu(float x) { return std::max(0.0f, x); }
inline float Tanh(float x) { return std::tanh(x); }
inline float Sin(float x) { return std::sin(x); }

// Activation Function
inline float Actf(float x, Activation act = NN_ACT) {
  switch (act) {
    case Activation::Sigmoid:
      return Sigmoid(x);
    case Activation::Relu:
      return Relu(x);
    case Activation::Tanh:
      return Tanh(x);
    case Activation::Sin:
      return Sin(x);
  }
  assert(false && "unreachable");
  return 0.0f;
}

// Derivative of the Activation Function
// y = activated output, z = pre-activation input
inline float Dactf(float y, float z, Activation dact = NN_ACT) {
  switch (dact) {
    case Activation::Sigmoid:
      return y * (1.0f - y);
    case Activation::Relu:
      return z > 0 ? 1.0f : NN_RELU_PARAM;
    case Activation::Tanh:
      return 1.0f - y * y;
    case Activation::Sin:
      return std::cos(z);
  }
  assert(false && "Unreachable");
  return 0.0f;
}

inline float rand_float(float low, float high) {
  static std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<float> dist(low, high);
  return dist(gen);
}

class Matrix {
 public:
  size_t rows;
  size_t cols;
  std::vector<float> data;

  Matrix(size_t r = 0, size_t c = 0, float d = 0.0f)
      : rows(r), cols(c), data(r * c, d) {}

  float& operator()(size_t i, size_t j) {
    assert(i < this->rows && j < this->cols);
    return this->data[i * cols + j];
  }

  // const one for reading only;
  const float& operator()(size_t i, size_t j) const {
    assert(i < rows && j < cols);
    return data[i * cols + j];
  }

  void fill(float x) { std::fill(data.begin(), data.end(), x); }

  void randomize(float low, float high) {
    for (auto& d : data) {
      d = rand_float(low, high);
    }
  }

  void apply_activation(Activation act) {
    for (auto& d : data) {
      d = Actf(d, act);
    }
  }

  Matrix& operator+=(const Matrix& other) {
    assert(other.cols == cols && rows == other.rows);
    for (size_t i = 0; i < data.size(); i++) {
      data[i] = data[i] + other.data[i];
    }

    return *this;
  }

  Matrix& operator*=(float scale) {
    assert(cols * rows == data.size());
    for (auto& d : data) {
      d *= scale;
    }
    return *this;
  }

  static Matrix dot(const Matrix& a, const Matrix& b) {
    assert(a.cols == b.rows);
    Matrix dst(a.rows, b.cols, 0.0f);
    for (size_t i = 0; i < dst.rows; ++i) {
      for (size_t j = 0; j < dst.cols; ++j) {
        for (size_t k = 0; k < a.cols; ++k) {
          dst(i, j) += a(i, k) * b(k, j);
        }
      }
    }
    return dst;
  }

  //  A faster version of dot() that uses std::thread
  static Matrix dot_mt(const Matrix& a, const Matrix& b) {
    assert(a.cols == b.rows);
    Matrix dst(a.rows, b.cols, 0.0f);

    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;  // Fallback just in case

    std::vector<std::thread> threads;

    size_t chunk_size = (dst.rows + num_threads - 1) / num_threads;

    for (unsigned int t = 0; t < num_threads; ++t) {
      size_t start_row = t * chunk_size;
      size_t end_row = std::min(start_row + chunk_size, dst.rows);

      if (start_row >= dst.rows) break;

      threads.emplace_back([start_row, end_row, &a, &b, &dst]() {
        for (size_t i = start_row; i < end_row; ++i) {
          //Loop order i -> k -> j is highly optimized for CPU Cache
          for (size_t k = 0; k < a.cols; ++k) {
            float a_ik = a(i, k);
            for (size_t j = 0; j < dst.cols; ++j) {
              dst(i, j) += a_ik * b(k, j);
            }
          }
        }
      });
    }

    for (auto& th : threads) {
      th.join();
    }

    return dst;
  }

  // row index , start col idx and how many next cols u want is that nums_cols
  Matrix slice_row(size_t row_idx, size_t start_col, size_t num_cols) const {
    assert(row_idx < rows);
    assert(start_col + num_cols <= cols);
    Matrix m(1, num_cols);
    for (size_t k = 0; k < num_cols; k++) {
      m(0, k) = (*this)(row_idx, start_col + k);
    }
    return m;
  }

  Matrix& transpose() {
    assert(rows * cols == data.size());
    std::vector<float> temp(data.size());
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        temp[j * rows + i] = data[i * cols + j];
      }
    }
    data = std::move(temp);
    std::swap(rows, cols);

    return *this;
  }

  Matrix inverse() const {
    if (rows != cols || rows == 0) {
      return Matrix();
    }

    const size_t n = rows;
    const float EPS = 1e-8f;

    // Augmented matrix [A | I]
    Matrix aug(n, 2 * n, 0.0f);

    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        aug(i, j) = (*this)(i, j);
      }
      aug(i, n + i) = 1.0f;
    }

    for (size_t p = 0; p < n; ++p) {
      size_t max_row = p;
      for (size_t i = p + 1; i < n; ++i) {
        if (std::abs(aug(i, p)) > std::abs(aug(max_row, p))) {
          max_row = i;
        }
      }

      if (max_row != p) {
        for (size_t j = 0; j < 2 * n; ++j) {
          std::swap(aug(p, j), aug(max_row, j));
        }
      }

      if (std::abs(aug(p, p)) < EPS) {
        return Matrix();  // not invertible
      }

      float pivot = aug(p, p);
      for (size_t j = 0; j < 2 * n; ++j) {
        aug(p, j) /= pivot;
      }

      for (size_t i = 0; i < n; ++i) {
        if (i != p) {
          float factor = aug(i, p);
          for (size_t j = 0; j < 2 * n; ++j) {
            aug(i, j) -= factor * aug(p, j);
          }
        }
      }
    }

    Matrix inv(n, n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        inv(i, j) = aug(i, n + j);
      }
    }

    return inv;
  }
  void print(const std::string& name, size_t padding = 0) const {
    std::string pad(padding, ' ');
    std::cout << pad << name << " = [\n";
    for (size_t i = 0; i < rows; ++i) {
      std::cout << pad << "    ";
      for (size_t j = 0; j < cols; ++j) {
        std::cout << (*this)(i, j) << " ";
      }
      std::cout << "\n";
    }
    std::cout << pad << "]\n";
  }
};

class NeuralNetwork {
 public:
  std::vector<size_t>
      arch;                // Architecture it stores number of neuros per layer
  std::vector<Matrix> ws;  // Weights
  std::vector<Matrix> bs;  // Biases
  std::vector<Matrix> as;  // Activations
  std::vector<Matrix> zs;  // Pre-activations (before activation function)

  NeuralNetwork(const std::vector<size_t>& architecture) : arch(architecture) {
    assert(arch.size() > 0);

    as.emplace_back(1, arch[0]);  // input layer for example if arch is {2 , 3 ,
                                  // 1} then input matix should be 1x2
    for (size_t i = 1; i < arch.size(); ++i) {
      ws.emplace_back(arch[i - 1], arch[i]);
      bs.emplace_back(1, arch[i]);
      as.emplace_back(1, arch[i]);
      zs.emplace_back(1, arch[i]);
    }
  }

  Matrix& get_input() { return as.front(); }
  Matrix& get_output() { return as.back(); }

  const Matrix& get_output() const { return as.back(); }
  void zero() {
    for (auto& a : as) {
      a.fill(0.0f);
    }
    for (auto& w : ws) {
      w.fill(0.0f);
    }
    for (auto& b : bs) {
      b.fill(0.0f);
    }
    for (auto& z : zs) {
      z.fill(0.0f);
    }
  }

  void randomize(float low, float high) {
    for (auto& a : as) {
      a.randomize(low, high);
    }
    for (auto& w : ws) {
      w.randomize(low, high);
    }
    for (auto& b : bs) {
      b.randomize(low, high);
    }
  }

  bool save(const std::string& filepath) const {
    std::ofstream outFile(filepath, std::ios::binary);
    if (!outFile.is_open()) {
      std::cout << filepath << " wasnt created some error" << std::endl;
      return false;
    }

    size_t arch_size = arch.size();

    outFile.write(reinterpret_cast<const char*>(&arch_size), sizeof(arch_size));

    outFile.write(reinterpret_cast<const char*>(arch.data()),
                  arch.size() * sizeof(size_t));

    for (const auto& w : ws) {
      outFile.write(reinterpret_cast<const char*>(&w.rows), sizeof(w.rows));
      outFile.write(reinterpret_cast<const char*>(&w.cols), sizeof(w.cols));
      outFile.write(reinterpret_cast<const char*>(w.data.data()),
                    w.data.size() * sizeof(float));
    }

    for (const auto& b : bs) {
      outFile.write(reinterpret_cast<const char*>(&b.rows), sizeof(b.rows));
      outFile.write(reinterpret_cast<const char*>(&b.cols), sizeof(b.cols));
      outFile.write(reinterpret_cast<const char*>(b.data.data()),
                    b.data.size() * sizeof(float));
    }

    outFile.close();
    return outFile.good();
  }

  bool load(const std::string& filepath) {
    std::ifstream inputFile(filepath, std::ios::binary);
    if (!inputFile.is_open()) {
      std::cout << filepath << " could not be opened" << std::endl;
      return false;
    }

    size_t arch_size = 0;
    if (!inputFile.read(reinterpret_cast<char*>(&arch_size),
                        sizeof(arch_size)) ||
        arch_size == 0) {
      return false;
    }

    std::vector<size_t> loaded_arch(arch_size);
    if (!inputFile.read(reinterpret_cast<char*>(loaded_arch.data()),
                        loaded_arch.size() * sizeof(size_t))) {
      return false;
    }

    std::vector<Matrix> loaded_ws;
    std::vector<Matrix> loaded_bs;
    std::vector<Matrix> loaded_as;
    std::vector<Matrix> loaded_zs;

    loaded_as.emplace_back(1, loaded_arch[0]);
    for (size_t i = 1; i < loaded_arch.size(); ++i) {
      loaded_ws.emplace_back(loaded_arch[i - 1], loaded_arch[i]);
      loaded_bs.emplace_back(1, loaded_arch[i]);
      loaded_as.emplace_back(1, loaded_arch[i]);
      loaded_zs.emplace_back(1, loaded_arch[i]);
    }

    for (auto& w : loaded_ws) {
      size_t rows = 0;
      size_t cols = 0;
      if (!inputFile.read(reinterpret_cast<char*>(&rows), sizeof(rows)) ||
          !inputFile.read(reinterpret_cast<char*>(&cols), sizeof(cols))) {
        return false;
      }
      if (rows != w.rows || cols != w.cols) {
        return false;
      }
      if (!inputFile.read(reinterpret_cast<char*>(w.data.data()),
                          w.data.size() * sizeof(float))) {
        return false;
      }
    }

    for (auto& b : loaded_bs) {
      size_t rows = 0;
      size_t cols = 0;
      if (!inputFile.read(reinterpret_cast<char*>(&rows), sizeof(rows)) ||
          !inputFile.read(reinterpret_cast<char*>(&cols), sizeof(cols))) {
        return false;
      }
      if (rows != b.rows || cols != b.cols) {
        return false;
      }
      if (!inputFile.read(reinterpret_cast<char*>(b.data.data()),
                          b.data.size() * sizeof(float))) {
        return false;
      }
    }

    arch = std::move(loaded_arch);
    ws = std::move(loaded_ws);
    bs = std::move(loaded_bs);
    as = std::move(loaded_as);
    zs = std::move(loaded_zs);

    return true;
  }

  void print(const std::string& name = "nn") const {
    std::cout << name << " = [\n";
    for (size_t i = 0; i < ws.size(); ++i) {
      ws[i].print("ws" + std::to_string(i), 4);
      bs[i].print("bs" + std::to_string(i), 4);
    }
    std::cout << "]\n";
  }

  void forward(Activation hidden_act = NN_ACT,
               Activation output_act = Activation::Sigmoid) {
    for (size_t i = 0; i < ws.size(); i++) {
      as[i + 1] =
          Matrix::dot_mt(as[i], ws[i]);  // matrix multiplicaton of weight and as
      as[i + 1] += bs[i];
      zs[i] = as[i + 1];  // save pre-activation values for backprop
      Activation layer_act = (i == ws.size() - 1) ? output_act : hidden_act;
      as[i + 1].apply_activation(layer_act);
    }
  }

  float cost(const Matrix& t, Activation hidden_act = NN_ACT,
             Activation output_act = Activation::Sigmoid) {
    assert(get_input().cols + get_output().cols == t.cols);
    float c = 0.0f;
    size_t n = t.rows;
    for (size_t i = 0; i < n; i++) {
      // we need to get output the true value
      Matrix inputs = t.slice_row(i, 0, get_input().cols);
      Matrix true_vals = t.slice_row(i, get_input().cols, get_output().cols);

      get_input() = inputs;
      forward(hidden_act, output_act);

      for (size_t j = 0; j < true_vals.cols; ++j) {
        float d = get_output()(0, j) - true_vals(0, j);
        c += d * d;
      }
    }
    return c / n;
  }

  NeuralNetwork backprop(const Matrix& t, Activation hidden_act = NN_ACT,
                         Activation output_act = Activation::Sigmoid) {
    size_t n = t.rows;
    assert(get_input().cols + get_output().cols == t.cols);

    NeuralNetwork g(arch);
    g.zero();

    for (size_t i = 0; i < n; ++i) {
      Matrix in = t.slice_row(i, 0, get_input().cols);

      Matrix out = t.slice_row(i, get_input().cols, get_output().cols);

      get_input() = in;

      forward(hidden_act, output_act);

      for (auto& a : g.as) {
        a.fill(0.0f);
      }

      for (size_t j = 0; j < out.cols; ++j) {
#ifdef NN_BACKPROP_TRADITIONAL
        g.get_output()(0, j) = 2.0f * (get_output()(0, j) - out(0, j));
#else
        g.get_output()(0, j) = get_output()(0, j) - out(0, j);
#endif
      }

#ifdef NN_BACKPROP_TRADITIONAL
      float s = 1.0f;
#else
      float s = 2.0f;
#endif

      for (size_t l = arch.size() - 1; l > 0; --l) {
        for (size_t j = 0; j < as[l].cols; ++j) {
          float a = as[l](0, j);
          float da = g.as[l](0, j);
          float z = zs[l - 1](0, j);  // pre-activation value
          Activation layer_act =
              (l == arch.size() - 1) ? output_act : hidden_act;
          float qa = Dactf(a, z, layer_act);

          g.bs[l - 1](0, j) += s * da * qa;

          for (size_t k = 0; k < as[l - 1].cols; ++k) {
            float pa = as[l - 1](0, k);
            float w = ws[l - 1](k, j);

            g.ws[l - 1](k, j) += s * da * qa * pa;
            g.as[l - 1](0, k) += s * da * qa * w;
          }
        }
      }
    }

    for (size_t i = 0; i < g.ws.size(); ++i) {
      for (size_t j = 0; j < g.ws[i].data.size(); ++j) {
        g.ws[i].data[j] /= n;
      }
      for (size_t j = 0; j < g.bs[i].data.size(); ++j) {
        g.bs[i].data[j] /= n;
      }
    }

    return g;
  }
  void learn(const NeuralNetwork& g, float rate) {
    for (size_t i = 0; i < ws.size(); ++i) {
      for (size_t j = 0; j < ws[i].data.size(); ++j) {
        ws[i].data[j] -= rate * g.ws[i].data[j];
      }
      for (size_t j = 0; j < bs[i].data.size(); ++j) {
        bs[i].data[j] -= rate * g.bs[i].data[j];
      }
    }
  }
};

struct Batch {
  size_t begin = 0;
  float cost = 0.0f;
  bool finished = false;

  void process(size_t batch_size, NeuralNetwork& nn, const Matrix& t,
               float rate, Activation hidden_act = NN_ACT,
               Activation output_act = Activation::Sigmoid) {
    if (finished) {
      finished = false;
      begin = 0;
      cost = 0.0f;
    }

    size_t size = batch_size;
    // for when batch_size gets greater than number of samples left
    if (begin + batch_size >= t.rows) {
      size = t.rows - begin;
    }

    // number of cols remain same coz u know inputs and outpus but we like make
    // small batchs of rows
    Matrix batch_t(size, t.cols);
    for (size_t i = 0; i < size; ++i) {
      for (size_t j = 0; j < t.cols; ++j) {
        batch_t(i, j) = t(begin + i, j);
      }
    }

    NeuralNetwork g = nn.backprop(batch_t, hidden_act, output_act);
    nn.learn(g, rate);
    cost += nn.cost(batch_t, hidden_act, output_act);
    begin += batch_size;

    if (begin >= t.rows) {
      size_t batch_count = (t.rows + batch_size - 1) / batch_size;
      cost /= batch_count;
      finished = true;
    }
  }
};

}  // namespace nn
