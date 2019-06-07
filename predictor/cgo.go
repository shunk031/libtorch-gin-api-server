package predictor

// #cgo CXXFLAGS: -std=c++11 -I${SRCDIR}/cbits -O3 -Wall -g -Wno-sign-compare -Wno-unused-function  -I/opt/libtorch/include -I/opt/libtorch/include/torch/csrc/api/include
// #cgo LDFLAGS: -lstdc++ -L/opt/libtorch/lib -ltorch -lcaffe2 -lc10 -lgomp
import "C"
