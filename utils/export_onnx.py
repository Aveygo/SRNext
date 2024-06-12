import torch
from archs.models.unext import uNextPrototype

generator = uNextPrototype()
generator.load_state_dict(torch.load("ckpts/uNextPrototype_Bootstrap_Latest.ckpt"), strict=True)

x = torch.randn(1, 3, 64, 64, requires_grad=False)
torch_out = generator(x)

# Export the model
torch.onnx.export(generator,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "model.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})