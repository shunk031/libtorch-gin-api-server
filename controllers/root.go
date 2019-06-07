package controllers

import (
	"github.com/gin-gonic/gin"
	"github.com/shunk031/libtorch-gin-api-server/domains"
)

func GetIndexHTML(c *gin.Context) {
	c.File(domains.GinIndexHTML)
}
