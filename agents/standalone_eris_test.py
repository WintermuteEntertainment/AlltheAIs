import torch
from eris_agent import Agent
from Eris import load_model, ERISEngine
import logging

logging.basicConfig(level=logging.INFO)

def test_eris_load():
    print("Starting Eris model load test...")
    
    try:
        print("Step 1: Loading tokenizer and base model...")
        tokenizer, base_model = load_model()
        print("✅ Tokenizer and base model loaded successfully")
        
        print("Step 2: Creating ERISEngine...")
        model = ERISEngine(tokenizer, base_model)
        print("✅ ERISEngine created successfully")
        
        if torch.cuda.is_available():
            print("Step 3: Moving model to GPU...")
            model = model.to(torch.device("cuda"))  # Use our custom to() method
            print("✅ Model moved to GPU")
        else:
            print("ℹ️ CUDA not available, using CPU")
            
        print("Step 4: Testing model inference...")
        test_output = model.generate_response("Test input")
        print(f"✅ Test output: {test_output[:50]}...")
        
        print("🎉 All tests passed successfully!")
    except Exception as e:
        print(f"❌ Error occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_eris_load()