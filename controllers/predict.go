package controllers

import (
	"bytes"
	"image"
	"log"

	"github.com/gin-gonic/gin"
	"github.com/nfnt/resize"
	"github.com/olahol/go-imageupload"
	"github.com/oliamb/cutter"
	"github.com/shunk031/libtorch-gin-api-server/domains"
	"github.com/shunk031/libtorch-gin-api-server/helpers"
	"github.com/shunk031/libtorch-gin-api-server/predictor"
)

func preprocessImg(img image.Image) (image.Image, error) {
	img = helpers.ResizeTransform(img, domains.ResizeSize)
	img, err := cutter.Crop(img, cutter.Config{
		Height: domains.QuadrateSize,
		Width:  domains.QuadrateSize,
		Mode:   cutter.Centered,
	})
	if err != nil {
		return nil, err
	}
	img = resize.Resize(domains.InputSize, domains.InputSize, img, resize.Lanczos3)
	return img, nil
}

func loadImgFromRequest(c *gin.Context) (image.Image, error) {
	img, err := imageupload.Process(c.Request, "file")
	if err != nil {
		return nil, err
	}
	decodedImg, _, err := image.Decode(bytes.NewReader(img.Data))
	if err != nil {
		return nil, err
	}
	return decodedImg, nil
}

// PredictProba ...
func PredictProba(c *gin.Context, model *predictor.Predictor, categories map[int][]string) {
	img, err := loadImgFromRequest(c)
	if err != nil {
		log.Fatal(err)
	}
	img, err = preprocessImg(img)
	if err != nil {
		log.Fatal(err)
	}
	if err = model.PredictProba(img); err != nil {
		log.Fatal(err)
	}
	pred, err := model.GetPrediction()
	if err != nil {
		log.Fatal(err)
	}
	maxIdx := helpers.Argmax(pred)
	c.JSON(200, domains.PredictResult{
		Desc:  categories[maxIdx][1],
		Score: pred[maxIdx],
	})
}
