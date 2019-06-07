package main

import (
	"log"

	"github.com/gin-gonic/gin"
	"github.com/shunk031/libtorch-gin-api-server/controllers"
	"github.com/shunk031/libtorch-gin-api-server/domains"
	"github.com/shunk031/libtorch-gin-api-server/helpers"
	"github.com/shunk031/libtorch-gin-api-server/predictor"
)

func main() {
	model, err := predictor.NewPredictor(domains.ModelFile)
	if err != nil {
		log.Fatal(err)
	}
	defer model.DeletePredictor()

	categories, err := helpers.GetCategories()
	if err != nil {
		log.Fatal(err)
	}

	r := gin.Default()
	r.GET("/", controllers.GetIndexHTML)
	r.POST("/predict", func(c *gin.Context) {
		controllers.PredictProba(c, model, categories)
		// controllers.UploadImage(c)
	})
	r.Run(domains.GinServerPort)
}
