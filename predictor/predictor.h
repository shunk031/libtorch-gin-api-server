#ifndef __PREDICTOR_H__
#define __PREDICTOR_H__

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

  typedef void *pPredictor;
  pPredictor NewPredictor(char *modelFile, int width, int height, int channels);
  void PredictProba(pPredictor predictor, float *inputData);
  const float *GetPrediction(pPredictor predictor);
  void DeletePredictor(pPredictor predictor);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __PREDICTOR_H__
