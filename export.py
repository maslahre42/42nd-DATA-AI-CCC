"""Standalone ONNX Export script for Colony: One."""

import argparse
from pathlib import Path

import onnx
import torch

from model import ColonyNet


def export(weights_path: str | None = None, output_path: str = "model.onnx") -> None:
    model = ColonyNet()
    if weights_path and Path(weights_path).exists():
        print(f"Loading weights from {weights_path}...")
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
        print("Using initialized model weights...")

    model.eval()
    dummy_input = torch.randn(1, 16, 8, 8, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["policy", "value"],
        dynamic_axes={
            "observation": {0: "batch"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
        },
        dynamo=False,
    )

    # Tournament disallows Constant operator nodes; convert to initializers
    onnx_model = onnx.load(output_path)
    clean_nodes = []
    for node in onnx_model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    t = attr.t
                    t.name = node.output[0]
                    onnx_model.graph.initializer.append(t)
        else:
            clean_nodes.append(node)

    del onnx_model.graph.node[:]
    onnx_model.graph.node.extend(clean_nodes)
    onnx.save(onnx_model, output_path)

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Successfully exported {output_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None, help="Path to .pt model weights")
    parser.add_argument("--output", type=str, default="model.onnx", help="Path to output ONNX file")
    args = parser.parse_args()
    export(args.weights, args.output)
