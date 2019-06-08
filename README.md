# High-speed Deep learning API Server with Libtorch (C++) and Gin (Golang)

[![GitHub license](https://img.shields.io/github/license/shunk031/libtorch-gin-api-server.svg)](https://github.com/shunk031/libtorch-gin-api-server/blob/master/LICENSE)
![Golang 1.9](https://img.shields.io/badge/golang-1.9%2B-blue.svg)
![Python 3.7](https://img.shields.io/badge/python-3.7%2B-brightgreen.svg)

## Convert your model to torch script

```shell
$ python convert_to_torch_script_via_tracing.py
# converted model is saved to assets/model.pt
```

## How to Run

```shell
$ docker-compose up api
```

## Credits

Parts of the implementation is borrowed from [rai-project/go-pytorch](https://github.com/rai-project/go-pytorch).
