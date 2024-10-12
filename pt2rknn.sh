#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <model_file>"
    exit 1
fi

MODEL_FILE=$1
BASE_DIR=$(dirname "$0")
ONNX_FILE="${MODEL_FILE%.*}.onnx"

# 进入ultralytics目录并导出模型
cd "$BASE_DIR/ultralytics"
export PYTHONPATH=./
python ./ultralytics/engine/exporter.py "$MODEL_FILE"

if [ $? -ne 0 ]; then
    echo "Model export failed."
    exit 1
fi

# 进入onnx2rknn目录并转换模型
cd "../onnx2rknn"
python convert.py "$ONNX_FILE" rk3588 fp "${MODEL_FILE%.*}.rknn"

if [ $? -ne 0 ]; then
    echo "Model conversion failed."
    exit 1
fi

echo "Model conversion to RKNN completed successfully."