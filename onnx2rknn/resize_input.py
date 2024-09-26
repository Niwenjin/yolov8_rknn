import onnx
onnx_model = onnx.load("model/mobilenet_v4_conv_m_scf/mobilenet_v4_conv_m_scf-epoch=125-top1_accur=0.8851-202409010434.onnx")

print("============input============")
print(onnx_model.graph.input)

print("============output============")
print(onnx_model.graph.output)


onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1

onnx_model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = 1

print("============new input============")
print(onnx_model.graph.input)

print("============new output============")
print(onnx_model.graph.output)

onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, 'model/mobilenet_v4_conv_m_scf/mobilenet_v4_conv_m_scf-epoch=125-top1_accur=0.8851-202409010434-new.onnx')
print("模型已保存")