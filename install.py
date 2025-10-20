import subprocess
import sys

def run_command(command, description):
    print(f"\n=== {description} ===")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"\n❌ Failed to execute: {command}")
        sys.exit(result.returncode)
    print(f"✅ Done: {description}")

def main():
    # Step 1: Install PyTorch with CUDA 12.9
    torch_cmd = (
        "pip install torch torchvision torchaudio "
        "--index-url https://download.pytorch.org/whl/cu129"
    )
    run_command(torch_cmd, "Installing PyTorch with CUDA 12.9")

    # Step 2: Install remaining packages from requirements.txt
    req_cmd = "pip install -r requirements.txt"
    run_command(req_cmd, "Installing packages from requirements.txt")

if __name__ == "__main__":
    main()
