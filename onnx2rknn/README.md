# ONNX2RKNN

将 onnx 模型转换为 airockchip 支持的 rknn 模型。

安装[rknn_toolkit2](https://github.com/airockchip/rknn-toolkit2.git)环境，python 验证`from rknn.api import RKNN`安装成功。

执行 convert.py 脚本将 onnx 模型转换为 rknn 模型，用法：

```sh
python convert.py onnx_model_path rk3588 fp output_rknn_path
```

对于输入维度不符合的模型，执行 resize_input.py 脚本将输入维度转换后才能执行 convert.py。

对于 yolov8 模型，需要通过 airockchip 提供的[ultralytics 库](https://github.com/airockchip/ultralytics_yolov8)将 pt 模型转换为 onxx 模型，然后才能转换为 rknn 模型。
