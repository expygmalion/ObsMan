import xgboost as xgb
import sys

print(f"Python version: {sys.version}")
print(f"XGBoost version: {xgb.__version__}")

# Try to create a classifier with GPU configuration
try:
    # Create XGBoost classifier with GPU tree method
    clf = xgb.XGBClassifier(
        tree_method='gpu_hist',  # Use GPU for tree construction
        gpu_id=0                 # Use first GPU
    )
    print("XGBoost with GPU configuration created successfully")
except Exception as e:
    print(f"Error creating XGBoost with GPU configuration: {e}")

# Print available parameters
print("\nAvailable tree methods:")
params = xgb.XGBClassifier().get_params()
print(f"Default tree_method: {params.get('tree_method', 'Not specified')}")

# Try to get list of available tree methods
try:
    from xgboost import testing as tm
    print(f"XGBoost GPU Support: {tm.gpu_acceleration_available()}")
except Exception as e:
    print(f"Could not determine GPU support directly: {e}")
    
print("\nTo use GPU with XGBoost, set these parameters:")
print("tree_method='gpu_hist', gpu_id=0") 