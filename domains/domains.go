package domains

const (
	GinIndexHTML  = "assets/index.html"
	GinServerPort = ":5000"
)

const (
	ResizeSize         = 316
	QuadrateSize       = 316
	InputSize          = 224
	ModelFile          = "assets/model.pt"
	ImageNetClassIndex = "assets/imagenet_class_index.json"
)

type PredictResult struct {
	Desc  string  `json:"description"`
	Score float32 `json:"score"`
}
