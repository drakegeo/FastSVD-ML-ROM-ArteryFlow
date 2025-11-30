"""
Main script to process all velocity components (x, y, z) automatically.

This is the primary entry point for POD basis calculation.
It processes all three velocity components sequentially using TRAINING DATA ONLY
and saves POD basis matrices to the data/ folder structure.
Linear projection of data is done separately in another script.

Usage:
    python src/linear_projection/process_all_components.py
"""
import sys
from pathlib import Path

# Add project root to path to import modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Now import with absolute path
from src.linear_projection.truncated_SVD_update import process_component

def process_all_components():
    """
    Process all three velocity components (x, y, z) sequentially.
    """
    components = ['x', 'y', 'z']
    
    print("=" * 60)
    print("Processing all velocity components (x, y, z)")
    print("=" * 60)
    
    results = {}
    
    for comp in components:
        print(f"\n{'='*60}")
        print(f"Processing component: {comp.upper()}")
        print(f"{'='*60}\n")
        
        try:
            result = process_component(component=comp)
            results[comp] = result
            print(f"\n✓ Component {comp.upper()} processed successfully!")
        except Exception as e:
            print(f"\n✗ Error processing component {comp}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print("\n" + "=" * 60)
    print("Processing Summary")
    print("=" * 60)
    
    successful = [comp for comp in components if comp in results]
    failed = [comp for comp in components if comp not in results]
    
    if successful:
        print(f"\n✓ Successfully processed: {', '.join([c.upper() for c in successful])}")
        print("\nOutput files created:")
        print("  POD basis:")
        for comp in successful:
            print(f"    - data/POD_basis/Basis_u{comp}.npy")
        print("  SVD results:")
        for comp in successful:
            print(f"    - data/SVD_results/trunc_size_{comp}.npy")
            print(f"    - data/SVD_results/trunc_l2_{comp}.npy")
    
    if failed:
        print(f"\n✗ Failed to process: {', '.join([c.upper() for c in failed])}")
    
    print("\n" + "=" * 60)
    print("All processing complete!")
    print("=" * 60)

if __name__ == '__main__':
    process_all_components()
