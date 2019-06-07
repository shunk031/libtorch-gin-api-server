package predictor

// #include <stdio.h>
// #include <stdlib.h>
// #include "predictor.h"
import "C"
import (
	"image"
	"unsafe"

	"github.com/Unknwon/com"
	"github.com/pkg/errors"
	"github.com/shunk031/libtorch-gin-api-server/helpers"
)

const (
	inputWidth   = 224
	inputHeight  = 224
	inputChannel = 3
	OutputSize   = 1000
	BatchSize    = 1
)

type Predictor struct {
	predictor C.pPredictor
}

func NewPredictor(modelFile string) (*Predictor, error) {

	if !com.IsFile(modelFile) {
		return nil, errors.Errorf("file %s not found", modelFile)
	}
	return &Predictor{
		predictor: C.NewPredictor(
			C.CString(modelFile),
			C.int(inputWidth),
			C.int(inputHeight),
			C.int(inputChannel),
		),
	}, nil
}

func (p *Predictor) PredictProba(img image.Image) error {
	data, err := helpers.ConvertImageToArray(
		img, []float32{0.485, 0.456, 0.406}, []float32{0.229, 0.224, 0.225})
	if err != nil {
		return err
	}
	ptr := (*C.float)(unsafe.Pointer(&data[0]))
	C.PredictProba(p.predictor, ptr)
	return nil
}

func (p *Predictor) GetPrediction() ([]float32, error) {
	cPrediction := C.GetPrediction(p.predictor)
	if cPrediction == nil {
		return nil, errors.Errorf("Empty prediction")
	}
	length := OutputSize * BatchSize
	slice := (*[1 << 30]float32)(unsafe.Pointer(cPrediction))[:length:length]
	return slice, nil
}

func (p *Predictor) DeletePredictor() {
	C.DeletePredictor(p.predictor)
}
