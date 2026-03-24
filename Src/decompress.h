#pragma once

#include "../Headers/nn.h"
#include "image_io.h"

Image ReconstructImage(nn::NeuralNetwork& compressor, int width, int height);
