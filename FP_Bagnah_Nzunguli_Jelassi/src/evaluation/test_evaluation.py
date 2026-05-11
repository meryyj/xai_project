# src/evaluation/test_evaluation.py

"""
Quick tests to verify evaluation modules work correctly.
Run with: python -m src.evaluation.test_evaluation
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add repository root to path so direct execution also works.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.evaluate import evaluate_model
from src.evaluation.confusion import compute_confusion_matrix, analyze_confusion_pairs
from src.evaluation.grad_cam import GradCAM, overlay_heatmap_on_image
from src.data.datasets import EMOTION_LABELS


def test_confusion_matrix():
    """Test confusion matrix computation"""
    print("\n" + "="*70)
    print("TEST: Confusion Matrix")
    print("="*70)
    
    # Dummy data
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2, 1, 1, 2])  # Some errors
    
    cm = compute_confusion_matrix(y_true, y_pred)
    print(f"Confusion matrix shape: {cm.shape}")
    print(f"Confusion matrix:\n{cm}")
    
    # Normalized
    cm_norm = compute_confusion_matrix(y_true, y_pred, normalize='true')
    print(f"\nNormalized confusion matrix:\n{cm_norm}")
    
    # Analyze pairs
    pairs_df = analyze_confusion_pairs(cm, ["class0", "class1", "class2"], top_k=5)
    print(f"\nConfused pairs:\n{pairs_df}")
    
    print("✅ Confusion matrix test passed!")
    return True


def test_gradcam():
    """Test Grad-CAM implementation"""
    print("\n" + "="*70)
    print("TEST: Grad-CAM")
    print("="*70)
    
    # Create dummy model
    class DummyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(32, 7)
        
        def forward(self, x):
            x = self.features(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = DummyCNN()
    target_layer = model.features[2]  # Last conv layer
    
    # Create Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # Dummy input
    dummy_input = torch.randn(1, 1, 64, 64)
    
    # Generate heatmap
    heatmap = grad_cam(dummy_input, target_class=0)
    
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Heatmap range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
    print(f"Heatmap mean: {heatmap.mean():.4f}")
    
    # Test overlay
    dummy_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    overlayed = overlay_heatmap_on_image(dummy_image, heatmap, alpha=0.5)
    
    print(f"Overlayed image shape: {overlayed.shape}")
    print(f"Overlayed image dtype: {overlayed.dtype}")
    
    print("✅ Grad-CAM test passed!")
    return True


def test_metrics_computation():
    """Test metrics computation"""
    print("\n" + "="*70)
    print("TEST: Metrics Computation")
    print("="*70)
    
    # Dummy predictions
    n_samples = 100
    n_classes = 7
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = y_true.copy()
    
    # Introduce some errors (20% error rate)
    n_errors = int(n_samples * 0.2)
    error_indices = np.random.choice(n_samples, n_errors, replace=False)
    for idx in error_indices:
        # Predict different class
        y_pred[idx] = (y_true[idx] + 1) % n_classes
    
    from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
    
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    
    # Should be around 0.8 (80% correct)
    assert 0.75 < acc < 0.85, f"Accuracy {acc} not in expected range"
    
    print("✅ Metrics computation test passed!")
    return True


def test_class_weights():
    """Test class weights computation"""
    print("\n" + "="*70)
    print("TEST: Class Weights")
    print("="*70)
    
    # Imbalanced class counts
    class_counts = [100, 200, 50, 150, 80, 120, 90]
    
    from src.training.losses import compute_class_weights
    weights = compute_class_weights(class_counts)
    
    print(f"Class counts: {class_counts}")
    print(f"Computed weights: {weights.numpy()}")
    
    # Verify inverse frequency property
    # Classes with fewer samples should have higher weights
    assert weights[2] > weights[1], "Minority class should have higher weight"
    
    print("✅ Class weights test passed!")
    return True


def test_evaluation_pipeline():
    """Test complete evaluation pipeline (without actual model)"""
    print("\n" + "="*70)
    print("TEST: Evaluation Pipeline")
    print("="*70)
    
    # This test verifies the structure is correct
    from src.evaluation.evaluate import save_results_to_csv
    
    # Dummy results
    results = {
        "accuracy": 0.85,
        "balanced_accuracy": 0.82,
        "macro_f1": 0.80,
        "weighted_f1": 0.83,
        "per_class_metrics": {
            emotion: {
                "precision": 0.8,
                "recall": 0.75,
                "f1": 0.77,
                "support": 100,
            }
            for emotion in EMOTION_LABELS
        },
        "y_true": np.random.randint(0, 7, 50),
        "y_pred": np.random.randint(0, 7, 50),
        "y_probs": np.random.rand(50, 7),
    }
    
    # Try to save (to temp directory)
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        save_results_to_csv(results, tmpdir, split_name="test")
        
        # Check files were created
        assert Path(tmpdir, "test_overall_metrics.csv").exists()
        assert Path(tmpdir, "test_per_class_metrics.csv").exists()
        assert Path(tmpdir, "test_predictions.csv").exists()
        
        print(f"✓ Results saved successfully to {tmpdir}")
    
    print("✅ Evaluation pipeline test passed!")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("RUNNING ALL EVALUATION TESTS")
    print("="*70)
    
    tests = [
        ("Confusion Matrix", test_confusion_matrix),
        ("Grad-CAM", test_gradcam),
        ("Metrics Computation", test_metrics_computation),
        ("Class Weights", test_class_weights),
        ("Evaluation Pipeline", test_evaluation_pipeline),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Test '{name}' FAILED with error:")
            print(f"   {type(e).__name__}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name:<30} {status}")
    
    n_passed = sum(1 for _, s in results if s)
    n_total = len(results)
    
    print(f"\nTotal: {n_passed}/{n_total} tests passed")
    
    if n_passed == n_total:
        print("\n🎉 ALL TESTS PASSED!")
        return True
    else:
        print(f"\n⚠️  {n_total - n_passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
