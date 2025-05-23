echo "Converting SAM2-Hiera-Tiny models..."
python -m export_sam2_camera --checkpoint checkpoints/sam2.1_hiera_tiny.pt \
    --output_parameters output_models/sam2.1_hiera_tiny.parameters.onnx \
    --output_encoder output_models/sam2.1_hiera_tiny.encoder.onnx \
    --output_decoder output_models/sam2.1_hiera_tiny.decoder.onnx \
    --output_memory_encoder output_models/sam2.1_hiera_tiny.memoryencoder.onnx \
    --output_memory_attention output_models/sam2.1_hiera_tiny.memoryattention.onnx \
    --model_type sam2.1_hiera_tiny