# test_checkpoint.py
import torch
from pathlib import Path

checkpoint_path = "experiments/checkpoints/best_model_20epochs.pt"

# Ajouter weights_only=False
ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("✅ Checkpoint loaded successfully!")
print("\nCheckpoint keys:", ckpt.keys())
print("\nEpoch:", ckpt.get('epoch', 'N/A'))
print("Best metric:", ckpt.get('best_metric', 'N/A'))

# Vérifier la structure du model_state
state_key = 'model_state' if 'model_state' in ckpt else 'model_state_dict'
if state_key in ckpt:
    model_state = ckpt[state_key]
    print(f"\n✅ Found '{state_key}' with {len(model_state)} keys")
    print(f"\nModel state keys (first 5):")
    for i, key in enumerate(list(model_state.keys())[:5]):
        print(f"  {key}: {model_state[key].shape}")
else:
    print("\n⚠️ Warning: No 'model_state' or 'model_state_dict' found!")
    print("Available keys:", list(ckpt.keys()))