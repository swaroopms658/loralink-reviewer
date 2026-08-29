import os
import sys
import shutil
import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer
from datasets import load_dataset
import gc
import json

# Define models to download
MODELS = {
    "gpt_neo": {
        "name": "EleutherAI/gpt-neo-2.7B",
        "description": "GPT-Neo 2.7B"
    },
    #"llama": {
    #    "name": "unsloth/Llama-3.2-3B",
    #    "description": "Llama 3.2 3B (Ungated community version)"
    #}
    #"qwen2.5": {
    #    "name": "Qwen/Qwen2.5-3B",
    #    "description": "Qwen2.5 3B"
    #},
    "phi": {
        "name": "microsoft/phi-1_5",
        "description": "Phi-1.5 1.3B (Reasoning/Code Lightweight)"
    }
}

# Define datasets to download
DATASETS = {
    "wikitext": {
        "name": "wikitext",
        "config": "wikitext-2-raw-v1",
        "description": "WikiText-2 (raw)"
    },
    "dolly": {
        "name": "databricks/databricks-dolly-15k",
        "config": None,
        "description": "Databricks Dolly 15k"
    },
    "e2e_nlg": {
        "name": "GEM/e2e_nlg",
        "config": None,
        "description": "E2E NLG Challenge",
        "trust_remote_code": True
    }
}

LOCAL_DATASET_DIR = "./dataset"


def download_model(model_name, description):
    """Download a single model from HuggingFace"""
    local_dir = os.path.join("./models", model_name)
    
    print(f"\n{'='*60}")
    print(f"📦 {description}")
    print(f"   Model: {model_name}")
    print(f"{'='*60}")
    
    try:
        if os.path.exists(os.path.join(local_dir, "config.json")):
            print(f"✓ Model already exists locally. Skipping download.")
            return local_dir
        
        print(f"Downloading model files...")
        print("This may take some time depending on model size...")
        
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            allow_patterns=["*.json", "*.txt", "*.md", "*.safetensors", "*.model", "*.tiktoken"],
            ignore_patterns=["*.bin", "*.msgpack", "*.ot", "*.h5", "*.gguf"]
        )
        
        print(f"✅ Downloaded to '{local_dir}'")
        return local_dir
        
    except Exception as e:
        print(f"❌ ERROR: Download failed for {model_name}")
        print(f"Details: {e}")
        return None


def verify_model(model_path, model_type):
    """Verify model can be loaded and get basic info"""
    print(f"\nVerifying model architecture...")
    
    try:
        config = AutoConfig.from_pretrained(model_path)
        print(f"  ✓ Model type: {config.model_type}")
        
        # Get number of layers (different attribute names for different architectures)
        if hasattr(config, 'num_layers'):
            num_layers = config.num_layers
        elif hasattr(config, 'num_hidden_layers'):
            num_layers = config.num_hidden_layers
        else:
            num_layers = "unknown"
        
        hidden_size = getattr(config, 'hidden_size', "unknown")
        vocab_size = getattr(config, 'vocab_size', "unknown")
        
        print(f"  ✓ Layers: {num_layers}")
        print(f"  ✓ Hidden size: {hidden_size}")
        print(f"  ✓ Vocab size: {vocab_size}")
        
        # Test tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            test_text = "Hello, world!"
            tokens = tokenizer.encode(test_text)
            print(f"  ✓ Tokenizer works ({len(tokens)} tokens for test text)")
        except Exception as e:
            print(f"  ⚠️ Tokenizer test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Verification failed: {e}")
        return False


def download_dataset(dataset_key, dataset_info):
    """Download a single dataset from HuggingFace"""
    ds_name = dataset_info["name"]
    ds_config = dataset_info.get("config")
    description = dataset_info["description"]

    label = f"{ds_name}/{ds_config}" if ds_config else ds_name
    print(f"\n{'='*60}")
    print(f"📊 Downloading Dataset: {label}")
    print(f"{'='*60}")

    try:
        args = {"path": ds_name, "cache_dir": LOCAL_DATASET_DIR}
        if ds_config:
            args["name"] = ds_config
        if dataset_info.get("trust_remote_code"):
            args["trust_remote_code"] = True
        load_dataset(**args)
        print(f"✅ Dataset '{label}' downloaded and cached to '{LOCAL_DATASET_DIR}'")
        return True

    except Exception as e:
        print(f"❌ ERROR: Dataset download failed for {label}")
        print(f"Details: {e}")
        return False


def generate_summary_report(results):
    """Generate a summary report of downloads"""
    print(f"\n\n{'='*60}")
    print("📋 DOWNLOAD SUMMARY REPORT")
    print(f"{'='*60}\n")
    
    successful = []
    failed = []
    
    for model_type, result in results.items():
        if result["success"]:
            successful.append(f"  ✅ {model_type.upper()}: {result['model_name']}")
        else:
            failed.append(f"  ❌ {model_type.upper()}: {result['model_name']}")
    
    if successful:
        print("Successfully Downloaded:")
        for line in successful:
            print(line)
    
    if failed:
        print("\nFailed Downloads:")
        for line in failed:
            print(line)
    
    print(f"\n{'='*60}")
    print(f"Total: {len(successful)} successful, {len(failed)} failed")
    print(f"{'='*60}\n")
    
    # Save report to file
    model_results = {k: v for k, v in results.items() if not k.startswith("dataset_")}
    dataset_results = {k: v for k, v in results.items() if k.startswith("dataset_")}
    report = {
        "successful_models": [r["model_name"] for r in model_results.values() if r["success"]],
        "failed_models": [r["model_name"] for r in model_results.values() if not r["success"]],
        "datasets": {k: {"name": v["model_name"], "downloaded": v["success"]} for k, v in dataset_results.items()}
    }
    
    with open("./download_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print("📄 Detailed report saved to: ./download_report.json\n")


def main():
    print("\n" + "="*60)
    print("🚀 LoraLink Multi-Model Downloader")
    print("   Downloading small test models for each architecture")
    print("="*60 + "\n")
    
    os.makedirs("./models", exist_ok=True)
    os.makedirs(LOCAL_DATASET_DIR, exist_ok=True)
    
    results = {}
    
    # Download each model
    for model_type, model_info in MODELS.items():
        model_path = download_model(model_info["name"], model_info["description"])
        
        if model_path:
            verified = verify_model(model_path, model_type)
            results[model_type] = {
                "model_name": model_info["name"],
                "success": verified,
                "path": model_path
            }
        else:
            results[model_type] = {
                "model_name": model_info["name"],
                "success": False,
                "path": None
            }
        
        # Clean up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Download datasets
    for ds_key, ds_info in DATASETS.items():
        dataset_success = download_dataset(ds_key, ds_info)
        label = f"{ds_info['name']}/{ds_info['config']}" if ds_info.get('config') else ds_info['name']
        results[f"dataset_{ds_key}"] = {
            "model_name": label,
            "success": dataset_success
        }
    
    # Generate summary
    generate_summary_report(results)
    
    # Show quick start instructions
    successful_models = [r for r in results.values() if r["success"] and r.get("path")]
    if successful_models:
        print("🎯 QUICK START EXAMPLES:\n")
        for model_type, model_info in MODELS.items():
            if results[model_type]["success"]:
                print(f"# Test with {model_type.upper()}:")
                print(f"python main.py --role coordinator --workers <ips> --host-ip <ip> --model-path {model_info['name']}\n")
        
        print("\n✅ All downloads complete! Ready for distributed training!\n")
    else:
        print("\n⚠️ No models were successfully downloaded. Please check errors above.\n")


if __name__ == "__main__":
    main()