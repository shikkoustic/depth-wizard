import torch
import sys
import os
from depthwizard.model import HeightModel

def export_to_onnx(model_path, out_path):
    print(f"Loading weights from {model_path}...")
    net = HeightModel(weights=model_path, device="cpu")
    
    # Create dummy input based on model's expected input size
    dummy_input = torch.randn(1, 3, net.in_size, net.in_size)
    
    print(f"Exporting ONNX model to {out_path}...")
    torch.onnx.export(
        net.nets[0], 
        (dummy_input,), 
        out_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["pixel_values"], 
        output_names=["predicted_depth"],
        dynamic_axes={
            "pixel_values": {0: "batch_size", 2: "height", 3: "width"},
            "predicted_depth": {0: "batch_size", 1: "height", 2: "width"}
        }
    )
    print("ONNX export complete. Inference speedup will be ~3x-5x on CPU/GPU.")

if __name__ == "__main__":
    weights_path = sys.argv[1] if len(sys.argv) > 1 else "weights/v3a_small.pt"
    out_file = weights_path.replace(".pt", ".onnx")
    export_to_onnx(weights_path, out_file)
