package helpers

import (
	"encoding/json"
	"image"
	"io/ioutil"
	"os"

	"github.com/nfnt/resize"
	"github.com/pkg/errors"
	"github.com/shunk031/libtorch-gin-api-server/domains"
)

// ResizeTransform ...
func ResizeTransform(img image.Image, size uint) image.Image {
	rct := img.Bounds()
	w := uint(rct.Dx())
	h := uint(rct.Dy())
	if (w <= h && w == size) || (h <= w && h == size) {
		return img
	}
	var ow, oh uint
	if w < h {
		ow = size
		oh = uint(size * h / w)
	} else {
		oh = size
		ow = uint(size * w / h)
	}
	return resize.Resize(ow, oh, img, resize.Lanczos3)
}

// ConvertImageToArray ...
func ConvertImageToArray(img image.Image, mean []float32, stddev []float32) ([]float32, error) {
	if img == nil {
		return nil, errors.Errorf("src image is nil")
	}

	bounds := img.Bounds()
	h := bounds.Max.Y - bounds.Min.Y // image height
	w := bounds.Max.X - bounds.Min.X // image width

	data := make([]float32, 3*h*w)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := img.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
			data[y*w+x] = ((float32(b>>8) / 255.0) - mean[0]) / stddev[0]
			data[w*h+y*w+x] = ((float32(g>>8) / 255.0) - mean[1]) / stddev[1]
			data[2*w*h+y*w+x] = ((float32(r>>8) / 255.0) - mean[2]) / stddev[2]
		}
	}
	return data, nil
}

// Argmax ...
func Argmax(pred []float32) int {
	r := 0
	for i := 0; i < len(pred); i++ {
		if pred[r] < pred[i] {
			r = i
		}
	}
	return r
}

func GetCategories() (map[int][]string, error) {
	reader, err := os.Open(domains.ImageNetClassIndex)
	if err != nil {
		return nil, err
	}
	defer reader.Close()

	catJSON, err := ioutil.ReadAll(reader)
	if err != nil {
		return nil, err
	}

	var categories map[int][]string
	err = json.Unmarshal(catJSON, &categories)
	if err != nil {
		return nil, err
	}
	return categories, err
}
