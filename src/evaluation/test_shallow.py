# test_shallow.py
import torch

checkpoint_path = "experiments/checkpoints/shallow_best.pt"
ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("✅ Shallow checkpoint loaded!")
print("\nCheckpoint keys:", ckpt.keys())
print("\nEpoch:", ckpt.get('epoch', 'N/A'))
print("Best metric (val_macro_f1):", ckpt.get('val_macro_f1', 'N/A'))

if 'model_state_dict' in ckpt:
    print(f"\n✅ Found 'model_state_dict' with {len(ckpt['model_state_dict'])} keys")