"""
One-time script to export SPECTER model to ONNX with INT8 quantization.

Run this locally (NOT on Render) to generate the quantized model:
    pip install sentence-transformers onnx onnxruntime
    python export_model.py

This will create model/model_quantized.onnx which should be committed to the repo.
"""

import os
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def export_to_onnx():
    print("Loading SPECTER model...")
    model = SentenceTransformer("allenai-specter")

    # Create model directory
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)

    # Get the transformer model
    transformer = model[0].auto_model
    tokenizer = model.tokenizer

    # Create dummy input for tracing
    dummy_text = "Sample text for model export"
    inputs = tokenizer(
        dummy_text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    token_type_ids = inputs.get("token_type_ids", torch.zeros_like(input_ids))

    # Export to ONNX using legacy exporter
    onnx_path = model_dir / "model.onnx"
    print(f"Exporting to ONNX: {onnx_path}")

    transformer.eval()
    with torch.no_grad():
        torch.onnx.export(
            transformer,
            (input_ids, attention_mask, token_type_ids),
            str(onnx_path),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["last_hidden_state", "pooler_output"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "token_type_ids": {0: "batch_size", 1: "sequence"},
                "last_hidden_state": {0: "batch_size", 1: "sequence"},
                "pooler_output": {0: "batch_size"},
            },
            opset_version=14,
            dynamo=False,  # Use legacy exporter for Windows compatibility
        )

    # Quantize the model
    print("Applying INT8 quantization...")
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantized_path = model_dir / "model_quantized.onnx"
    quantize_dynamic(
        str(onnx_path),
        str(quantized_path),
        weight_type=QuantType.QInt8,
    )

    # Remove the unquantized model to save space
    os.remove(onnx_path)

    # Verify the quantized model works
    print("Verifying quantized model...")
    import onnxruntime as ort

    session = ort.InferenceSession(str(quantized_path))

    # Test inference
    outputs = session.run(
        None,
        {
            "input_ids": input_ids.numpy(),
            "attention_mask": attention_mask.numpy(),
            "token_type_ids": token_type_ids.numpy(),
        }
    )

    print(f"Output shape: {outputs[0].shape}")

    # Check file sizes
    quantized_size = quantized_path.stat().st_size / (1024 * 1024)
    print(f"\nQuantized model size: {quantized_size:.1f} MB")
    print(f"Model saved to: {quantized_path}")

    # Save tokenizer config for reference
    print("\nTokenizer will be loaded from HuggingFace at runtime.")
    print("Model name: allenai-specter")


if __name__ == "__main__":
    export_to_onnx()
