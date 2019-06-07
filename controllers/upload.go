package controllers

import (
	"bytes"
	"fmt"
	"image"
	"image/png"
	"io/ioutil"
	"log"
	"net/http"
	"path/filepath"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/nfnt/resize"
	"github.com/olahol/go-imageupload"
	"github.com/oliamb/cutter"
	"github.com/shunk031/libtorch-gin-api-server/domains"
	"github.com/shunk031/libtorch-gin-api-server/helpers"
)

const (
	UploadedImagePath = "assets/uploaded"
)

func UploadImage(c *gin.Context) {
	img, err := imageupload.Process(c.Request, "file")
	if err != nil {
		log.Fatal(err)
	}

	decodedImg, format, err := image.Decode(bytes.NewReader(img.Data))
	if err != nil {
		log.Fatal(err)
	}

	bounds := decodedImg.Bounds()
	fmt.Println(format, bounds.Dx(), bounds.Dy())

	resizedImg := helpers.ResizeTransform(decodedImg, domains.ResizeSize)
	croppedImg, err := cutter.Crop(resizedImg, cutter.Config{
		Width:  domains.QuadrateSize,
		Height: domains.QuadrateSize,
		Mode:   cutter.Centered,
	})
	if err != nil {
		log.Fatal(err)
	}
	resizedImg = resize.Resize(domains.InputSize, domains.InputSize, croppedImg, resize.Lanczos3)

	data := new(bytes.Buffer)
	if err = png.Encode(data, resizedImg); err != nil {
		log.Fatal(err)
	}
	bs := data.Bytes()
	err = ioutil.WriteFile(
		filepath.Join(
			UploadedImagePath,
			fmt.Sprintf("%d.png", time.Now().Unix())), bs, 0600)
	if err != nil {
		log.Fatal(err)
	}
	c.Redirect(http.StatusMovedPermanently, "/")
}
