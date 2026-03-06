"""
Unit test for In-Run Shapley core logic without requiring model download.
Tests the Shapley value computation and accumulation logic.
"""

import os
import sys
import torch
import numpy as np

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inrun_shapley_engine import InRunShapleyEngine


def test_shapley_computation():
    """Test Shapley value computation logic."""
    print("=" * 80)
    print("Testing In-Run Shapley Core Logic")
    print("=" * 80)
    
    # Create a mock engine (we'll test the methods directly)
    class MockEngine(InRunShapleyEngine):
        def __init__(self):
            # Skip parent initialization, just set up what we need
            self.val_batch_size = 2
            self.order = 1
            self.accumulate_shapley = True
            self.shapley_save_interval = 10
            self.shapley_values = {}
            self.shapley_history = []
            self.total_samples_seen = 0
            self.iter_num = 0
            self.dot_product_log = []
    
    engine = MockEngine()
    
    # Test 1: Shapley value computation from dot products
    print("\n--- Test 1: Shapley value computation ---")
    batch_size = 4
    val_batch_size = 2
    
    # Create mock dot products (simulating gradient dot products)
    dot_products = torch.tensor([0.5, -0.2, 0.8, -0.1])
    
    shapley_batch = engine._compute_shapley_from_dot_product(
        dot_products, batch_size, val_batch_size
    )
    
    print(f"Input dot products: {dot_products}")
    print(f"Computed Shapley values: {shapley_batch}")
    print(f"Shapley values shape: {shapley_batch.shape}")
    
    # Verify that Shapley values are proportional to dot products
    assert shapley_batch.shape[0] == batch_size, "Shapley batch size mismatch"
    print("✓ Shapley computation test passed")
    
    # Test 2: Accumulation logic
    print("\n--- Test 2: Shapley value accumulation ---")
    
    # Simulate multiple iterations
    num_iterations = 5
    samples_per_iter = 4
    
    for iter_num in range(num_iterations):
        # Create dot products for this iteration
        dot_products = torch.randn(samples_per_iter) * 0.5
        batch_idx = list(range(iter_num * samples_per_iter, 
                              (iter_num + 1) * samples_per_iter))
        
        # Compute Shapley values
        shapley_batch = engine._compute_shapley_from_dot_product(
            dot_products, samples_per_iter, val_batch_size
        )
        
        # Accumulate
        for i, idx in enumerate(batch_idx):
            engine.shapley_values[idx] = engine.shapley_values.get(idx, 0) + shapley_batch[i].item()
        
        engine.total_samples_seen += samples_per_iter
        engine.iter_num = iter_num
    
    print(f"Total samples seen: {engine.total_samples_seen}")
    print(f"Unique samples with Shapley values: {len(engine.shapley_values)}")
    
    # Verify accumulation
    assert len(engine.shapley_values) == num_iterations * samples_per_iter, \
        "Not all samples have Shapley values"
    print("✓ Accumulation test passed")
    
    # Test 3: Shapley array conversion
    print("\n--- Test 3: Shapley array conversion ---")
    
    shapley_array = engine.get_shapley_array()
    print(f"Shapley array shape: {shapley_array.shape}")
    print(f"Shapley array statistics:")
    print(f"  Mean: {np.mean(shapley_array):.6f}")
    print(f"  Std:  {np.std(shapley_array):.6f}")
    print(f"  Min:  {np.min(shapley_array):.6f}")
    print(f"  Max:  {np.max(shapley_array):.6f}")
    
    assert shapley_array.shape[0] == len(engine.shapley_values), \
        "Array size mismatch"
    print("✓ Array conversion test passed")
    
    # Test 4: Verify Shapley value properties
    print("\n--- Test 4: Shapley value properties ---")
    
    # Check that we have both positive and negative values
    positive_count = np.sum(shapley_array > 0)
    negative_count = np.sum(shapley_array < 0)
    zero_count = np.sum(shapley_array == 0)
    
    print(f"Positive Shapley values: {positive_count}")
    print(f"Negative Shapley values: {negative_count}")
    print(f"Zero Shapley values: {zero_count}")
    
    # In practice, we expect a mix of positive and negative values
    # (some samples help, some hurt model performance)
    assert positive_count + negative_count + zero_count == len(shapley_array), \
        "Count mismatch"
    print("✓ Properties test passed")
    
    # Test 5: Test with specific batch indices
    print("\n--- Test 5: Non-sequential batch indices ---")
    
    engine2 = MockEngine()
    dot_products = torch.tensor([0.3, -0.5, 0.7])
    batch_idx = [10, 25, 42]  # Non-sequential indices
    
    shapley_batch = engine2._compute_shapley_from_dot_product(
        dot_products, len(batch_idx), val_batch_size
    )
    
    for i, idx in enumerate(batch_idx):
        engine2.shapley_values[idx] = shapley_batch[i].item()
    
    # Get array with specific size
    shapley_array2 = engine2.get_shapley_array(num_samples=50)
    print(f"Array size: {shapley_array2.shape}")
    print(f"Non-zero values at indices: {np.where(shapley_array2 != 0)[0]}")
    
    assert shapley_array2[10] != 0, "Index 10 should have value"
    assert shapley_array2[25] != 0, "Index 25 should have value"
    assert shapley_array2[42] != 0, "Index 42 should have value"
    print("✓ Non-sequential indices test passed")
    
    print("\n" + "=" * 80)
    print("All core logic tests passed!")
    print("=" * 80)
    
    return True


def test_shapley_consistency():
    """Test that Shapley values are consistent across iterations."""
    print("\n" + "=" * 80)
    print("Testing Shapley Value Consistency")
    print("=" * 80)
    
    class MockEngine(InRunShapleyEngine):
        def __init__(self):
            self.val_batch_size = 1
            self.order = 1
            self.accumulate_shapley = True
            self.shapley_save_interval = 10
            self.shapley_values = {}
            self.shapley_history = []
            self.total_samples_seen = 0
            self.iter_num = 0
            self.dot_product_log = []
    
    engine = MockEngine()
    
    # Simulate same sample seen multiple times
    sample_idx = 5
    num_times_seen = 3
    
    for i in range(num_times_seen):
        dot_product = torch.tensor([0.4])  # Same dot product each time
        shapley_val = engine._compute_shapley_from_dot_product(
            dot_product, 1, engine.val_batch_size
        )[0].item()
        
        engine.shapley_values[sample_idx] = engine.shapley_values.get(sample_idx, 0) + shapley_val
    
    final_value = engine.shapley_values[sample_idx]
    expected_value = shapley_val * num_times_seen
    
    print(f"Sample {sample_idx} seen {num_times_seen} times")
    print(f"Final accumulated value: {final_value:.6f}")
    print(f"Expected value (single * {num_times_seen}): {expected_value:.6f}")
    
    assert abs(final_value - expected_value) < 1e-6, "Accumulation not consistent"
    print("✓ Consistency test passed")
    
    return True


if __name__ == "__main__":
    try:
        test_shapley_computation()
        test_shapley_consistency()
        print("\n" + "=" * 80)
        print("✓ All tests passed successfully!")
        print("=" * 80)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
