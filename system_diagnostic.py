import torch
import logging
import platform
import sys
from agents.Eris import load_model, ERISEngine

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("system_diagnostic.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SystemDiagnostic")

def log_hardware_info():
    """Log detailed hardware information"""
    logger.info("\n===== Hardware Information =====")
    
    # CPU Information
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        logger.info(f"CPU: {info['brand_raw']}")
        logger.info(f"Cores: {info['count']}")
    except ImportError:
        logger.warning("py-cpuinfo not installed, skipping CPU details")
    
    # GPU Information
    logger.info("\n===== GPU Information =====")
    if torch.cuda.is_available():
        logger.info(f"CUDA available: Yes")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  Total Memory: {props.total_memory/1e9:.2f} GB")
            logger.info(f"  Memory allocated: {torch.cuda.memory_allocated(i)/1e6:.2f} MB")
            logger.info(f"  Memory reserved: {torch.cuda.memory_reserved(i)/1e6:.2f} MB")
    else:
        logger.info("CUDA not available")
    
    # RAM Information
    logger.info("\n===== Memory Information =====")
    try:
        import psutil
        mem = psutil.virtual_memory()
        logger.info(f"Total RAM: {mem.total/1e9:.2f} GB")
        logger.info(f"Available RAM: {mem.available/1e9:.2f} GB")
    except ImportError:
        logger.warning("psutil not installed, skipping memory details")

def log_environment_info():
    """Log environment and package information"""
    logger.info("\n===== Environment Information =====")
    logger.info(f"OS: {platform.system()} {platform.release()}")
    logger.info(f"Python: {sys.version}")
    
    logger.info("\n===== Installed Packages =====")
    try:
        from pip._internal.operations import freeze
        packages = freeze.freeze()
        for pkg in packages:
            if any(key in pkg.lower() for key in ["torch", "cuda", "tensor", "transform"]):
                logger.info(pkg)
    except ImportError:
        logger.warning("Could not get installed packages list")

def test_model_loading():
    """Test loading and running the Eris model"""
    logger.info("\n===== Model Loading Test =====")
    
    try:
        # Step 1: Load tokenizer and base model
        logger.info("Loading tokenizer and base model...")
        tokenizer, base_model = load_model()
        logger.info("✅ Tokenizer and base model loaded successfully")
        
        # Step 2: Create ERISEngine
        logger.info("Creating ERISEngine...")
        model = ERISEngine(tokenizer, base_model)
        logger.info("✅ ERISEngine created successfully")
        
        # Step 3: Move to GPU if available
        if torch.cuda.is_available():
            logger.info("Moving model to GPU...")
            try:
                model = model.to(torch.device("cuda"))
                logger.info("✅ Model moved to GPU")
            except Exception as e:
                logger.error(f"❌ GPU transfer failed: {type(e).__name__}: {e}")
                logger.info("Trying to run on CPU instead...")
                model = model.to(torch.device("cpu"))
        else:
            logger.info("Using CPU for model")
        
        # Step 4: Test inference
        logger.info("Testing model inference...")
        test_input = "The meaning of life is"
        logger.info(f"Input: '{test_input}'")
        test_output = model.generate_response(test_input)
        logger.info(f"Output: {test_output}")
        
        # Step 5: Verify output
        if test_output and len(test_output) > 0:
            logger.info("✅ Model inference successful!")
            logger.info(f"Generated text: {test_output[:100]}...")
        else:
            logger.error("❌ Model returned empty response")
            
    except Exception as e:
        logger.exception("❌ Critical error occurred during model loading")

def main():
    logger.info("Starting system diagnostic...")
    
    # Log environment and hardware info
    log_environment_info()
    log_hardware_info()
    
    # Test basic tensor operations
    logger.info("\n===== Tensor Operation Test =====")
    try:
        logger.info("Creating random tensor...")
        x = torch.rand(3, 3)
        logger.info(f"Tensor created: {x}")
        
        if torch.cuda.is_available():
            logger.info("Moving tensor to GPU...")
            x = x.cuda()
            logger.info(f"Tensor on GPU: {x}")
            logger.info("Performing GPU operation...")
            y = x * 2
            logger.info(f"Operation result: {y}")
        
        logger.info("✅ Tensor operations successful!")
    except Exception as e:
        logger.exception("❌ Tensor operations failed")
    
    # Test model loading
    test_model_loading()
    
    logger.info("\n===== Diagnostic Complete =====")

if __name__ == "__main__":
    main()