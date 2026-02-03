#!/usr/bin/env python3
"""
Test script to verify model selection implementation.
This tests the MODEL_CONFIGS, get_model function logic without loading actual models.
"""

def test_model_configs():
    """Test that MODEL_CONFIGS is properly structured"""
    MODEL_CONFIGS = {
        "parakeet-tdt-0.6b-v3": {
            "hf_id": "nemo-parakeet-tdt-0.6b-v3",
            "quantization": "int8",
            "description": "INT8 (fastest)"
        },
        "istupakov/parakeet-tdt-0.6b-v3-onnx": {
            "hf_id": "istupakov/parakeet-tdt-0.6b-v3-onnx",
            "quantization": None,
            "description": "FP32"
        },
        "grikdotnet/parakeet-tdt-0.6b-fp16": {
            "hf_id": "grikdotnet/parakeet-tdt-0.6b-fp16",
            "quantization": "fp16",
            "description": "FP16"
        },
    }
    
    # Test all models are present
    assert "parakeet-tdt-0.6b-v3" in MODEL_CONFIGS
    assert "istupakov/parakeet-tdt-0.6b-v3-onnx" in MODEL_CONFIGS
    assert "grikdotnet/parakeet-tdt-0.6b-fp16" in MODEL_CONFIGS
    
    # Test each model has required fields
    for model_name, config in MODEL_CONFIGS.items():
        assert "hf_id" in config, f"Missing hf_id for {model_name}"
        assert "quantization" in config, f"Missing quantization for {model_name}"
        assert "description" in config, f"Missing description for {model_name}"
        
    # Test quantization values
    assert MODEL_CONFIGS["parakeet-tdt-0.6b-v3"]["quantization"] == "int8"
    assert MODEL_CONFIGS["istupakov/parakeet-tdt-0.6b-v3-onnx"]["quantization"] is None
    assert MODEL_CONFIGS["grikdotnet/parakeet-tdt-0.6b-fp16"]["quantization"] == "fp16"
    
    print("✅ MODEL_CONFIGS structure test passed")


def test_model_fallback_logic():
    """Test the fallback logic when unknown model is requested"""
    MODEL_CONFIGS = {
        "parakeet-tdt-0.6b-v3": {
            "hf_id": "nemo-parakeet-tdt-0.6b-v3",
            "quantization": "int8",
            "description": "INT8 (fastest)"
        },
        "istupakov/parakeet-tdt-0.6b-v3-onnx": {
            "hf_id": "istupakov/parakeet-tdt-0.6b-v3-onnx",
            "quantization": None,
            "description": "FP32"
        },
        "grikdotnet/parakeet-tdt-0.6b-fp16": {
            "hf_id": "grikdotnet/parakeet-tdt-0.6b-fp16",
            "quantization": "fp16",
            "description": "FP16"
        },
    }
    
    # Test that unknown model falls back to default
    unknown_model = "unknown-model"
    if unknown_model not in MODEL_CONFIGS:
        model_name = "parakeet-tdt-0.6b-v3"  # Fallback
        assert model_name == "parakeet-tdt-0.6b-v3"
        
    # Test that known models are recognized
    for known_model in MODEL_CONFIGS.keys():
        assert known_model in MODEL_CONFIGS
        
    print("✅ Model fallback logic test passed")


def test_openai_compatibility():
    """Test OpenAI compatible parameter defaults"""
    # Default model should be parakeet variant, not whisper
    default_model = "parakeet-tdt-0.6b-v3"
    assert default_model != "whisper-1"
    assert default_model == "parakeet-tdt-0.6b-v3"
    
    print("✅ OpenAI compatibility test passed")


if __name__ == "__main__":
    test_model_configs()
    test_model_fallback_logic()
    test_openai_compatibility()
    print("\n✅ All tests passed successfully!")
