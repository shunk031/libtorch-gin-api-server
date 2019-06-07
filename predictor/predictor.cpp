#define _GLIBCXX_USE_CXX11_ABI 0

#include <torch/torch.h>
#include <torch/script.h>

#include "predictor.h"

class Predictor {
  int width, height, channels;
  std::shared_ptr<torch::jit::script::Module> model;
public:
  at::Tensor result;
  Predictor(const std::string &modelFile, int witdh, int height, int channels);
  void PredictProba(float *inputData);
};

Predictor::Predictor(const std::string &modelFile, int width, int height, int channels){
  this->width = width;
  this->height = height;
  this->channels = channels;

  model = torch::jit::load(modelFile);
  assert(model != nullptr);
}

void Predictor::PredictProba(float *inputData) {
  std::vector<int64_t> sizes = {1, 3, width, height};
  at::Tensor tensorImg = torch::from_blob(inputData, at::IntList(sizes));

  std::vector<torch::jit::IValue> inputs;
  inputs.emplace_back(tensorImg);
  result = at::softmax(model->forward(inputs).toTensor(), 1);
}

pPredictor NewPredictor(char *modelFile, int width, int height, int channels){
  try {
    const auto predictor = new Predictor(modelFile, width, height, channels);
    return (void *)predictor;

  } catch(const std::invalid_argument &ex) {
    return nullptr;
  }
}

void PredictProba(pPredictor predictor, float *inputData) {
  auto pred = (Predictor *)predictor;
  if (pred == nullptr) {
    return;
  }
  pred->PredictProba(inputData);
}

const float *GetPrediction(pPredictor predictor) {
  auto pred = (Predictor *)predictor;
  if (pred == nullptr) {
    return nullptr;
  }
  return pred->result.data<float>();
}

void DeletePredictor(pPredictor predictor) {
  auto pred = (Predictor *)predictor;
  if (pred == nullptr) {
    return;
  }
  delete pred;
}

