#include "compress.h"
#include "decompress.h"

#include <iostream>
#include <string>

namespace {

void PrintUsage(const char* program_name) {
  std::cout << "Usage:\n";
  std::cout << "  " << program_name << " compress [input_image] [output_ninc]\n";
  std::cout << "  " << program_name << " decompress [input_ninc] [output_png]\n";
}

} // namespace

int main(int argc, char** argv) {
  if (argc == 1) {
    return RunCompress("test.jpg", "test.ninc");
  }

  const std::string command = argv[1];
  if (command == "compress") {
    const std::string input_path = argc > 2 ? argv[2] : "test.jpg";
    const std::string model_path = argc > 3 ? argv[3] : "test.ninc";
    return RunCompress(input_path, model_path);
  }

  if (command == "decompress") {
    const std::string model_path = argc > 2 ? argv[2] : "test.ninc";
    const std::string output_path = argc > 3 ? argv[3] : "reconstructed.png";
    return RunDecompress(model_path, output_path);
  }

  PrintUsage(argv[0]);
  return 1;
}
