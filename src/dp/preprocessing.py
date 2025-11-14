from pathlib import Path
import pickle
from typing import Literal, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from utils.returnFormat import returnFormat

# Global state
label_encoders = {}
scaler = MinMaxScaler()
categorical_features = ['demand_7d', 'time_of_day', 'season']
numerical_features = ['current_price', 'rating_7d', 'orders_7d', 'revenue_7d', 'avg_quantity']
binary_features = ['is_weekend', 'is_holiday', 'is_event']
feature_stats = {}
is_fitted = False


def fit_preprocess(df: pd.DataFrame):
    """Fit the preprocessor on training data"""
    global is_fitted, label_encoders, scaler, feature_stats
    
    print("\nFitting preprocessor...")
    
    # Fit label encoders for categorical features
    for cf in categorical_features:
        if cf in df.columns:
            le = LabelEncoder()
            le.fit(df[cf].astype(str))
            label_encoders[cf] = le
            print(f"  {cf}: {len(le.classes_)} classes")
    
    # Fit scaler for numerical features
    numerical_cols = [col for col in numerical_features if col in df.columns]
    if numerical_cols:
        scaler.fit(df[numerical_cols])
        print(f"  Scaled {len(numerical_cols)} numerical features")
        
        # Store feature statistics
        for col in numerical_cols:
            feature_stats[col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'std': float(df[col].std())
            }
    
    is_fitted = True
    print("  Preprocessor fitted successfully!")


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Transform data using fitted preprocessor"""
    if not is_fitted:
        raise ValueError("Preprocessor must be fitted before transform")
    
    df = df.copy()
    
    # Transform categorical features
    for cf in categorical_features:
        if cf in df.columns and cf in label_encoders:
            df[cf] = label_encoders[cf].transform(df[cf].astype(str))
    
    # Transform numerical features
    numerical_cols = [col for col in numerical_features if col in df.columns]
    if numerical_cols:
        df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    # Convert binary features to int
    for bf in binary_features:
        if bf in df.columns:
            df[bf] = df[bf].astype(int)
    
    return df


def fit_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Fit and transform in one step"""
    fit_preprocess(df)
    return transform(df)


def inverse_transform_price(scaled_price: np.ndarray) -> np.ndarray:
    """Convert scaled price back to original scale"""
    if isinstance(scaled_price, (int, float)):
        scaled_price = [scaled_price]
    
    scaled_price = np.array(scaled_price).flatten()
    
    # Create dummy array with zeros for other features
    dummy = np.zeros((len(scaled_price), len(numerical_features)))
    dummy[:, 0] = scaled_price  # current_price is first feature
    
    # Inverse transform
    original = scaler.inverse_transform(dummy)
    return original[:, 0]


def save_preprocessor(filepath: Path):
    """Save preprocessor to disk"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    preprocessor_data = {
        'label_encoders': label_encoders,
        'scaler': scaler,
        'feature_stats': feature_stats,
        'categorical_features': categorical_features,
        'numerical_features': numerical_features,
        'binary_features': binary_features,
        'is_fitted': is_fitted
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(preprocessor_data, f)
    
    print(f"Preprocessor saved to {filepath}")


def load_preprocessor(filepath: Path):
    """Load preprocessor from disk"""
    global label_encoders, scaler, feature_stats, is_fitted
    global categorical_features, numerical_features, binary_features
    
    with open(filepath, 'rb') as f:
        preprocessor_data = pickle.load(f)
    
    label_encoders = preprocessor_data['label_encoders']
    scaler = preprocessor_data['scaler']
    feature_stats = preprocessor_data['feature_stats']
    categorical_features = preprocessor_data['categorical_features']
    numerical_features = preprocessor_data['numerical_features']
    binary_features = preprocessor_data['binary_features']
    is_fitted = preprocessor_data['is_fitted']
    
    print(f"Preprocessor loaded from {filepath}")


def prepare_sequences(
    df: pd.DataFrame, 
    sequence_length: int = 30,
    target_col: str = 'price_change_percent'
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare sequences for LSTM training"""
    
    if 'item_id' not in df.columns:
        raise ValueError("item_id column is required")
    
    # Define feature columns
    exclude_cols = [
        target_col, 'item_id', 'timestamp', 'dt', 'event_name', 
        'branch_id', 'predicted_price', 'base_price'
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"\nPreparing sequences with {len(feature_cols)} features:")
    print(f"  Features: {feature_cols}")
    
    X, y = [], []
    skipped = 0
    
    # Group by item_id to create sequences per item
    for item_id in df['item_id'].unique():
        item_data = df[df['item_id'] == item_id].sort_values('timestamp')
        
        # Skip items with insufficient data
        if len(item_data) < sequence_length + 1:
            skipped += 1
            continue
        
        item_features = item_data[feature_cols].values
        item_target = item_data[target_col].values
        
        # Create sliding windows
        for i in range(len(item_data) - sequence_length):
            X.append(item_features[i:i + sequence_length])
            y.append(item_target[i + sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    if skipped > 0:
        print(f"  Skipped {skipped} items with insufficient data")
    
    print(f"  Created {len(X)} sequences")
    print(f"  Shape: X={X.shape}, y={y.shape}")
    
    return X, y


def calculate_price_change_percent(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate price change percentage from base price"""
    df = df.copy()
    
    # Ensure base_price exists
    if 'base_price' not in df.columns:
        df['base_price'] = df.groupby('item_id')['current_price'].transform('first')
    
    # Calculate percentage change
    df['price_change_percent'] = (
        (df['current_price'] - df['base_price']) / df['base_price']
    ) * 100
    
    # Clip to reasonable bounds
    df['price_change_percent'] = df['price_change_percent'].clip(-50, 100)
    
    return df


def calculate_demand_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate rolling demand metrics"""
    df = df.copy()
    df = df.sort_values(['item_id', 'timestamp'])
    
    # If orders_7d doesn't exist, create a proxy
    if 'orders_7d' not in df.columns:
        df['orders_7d'] = 0  # Will be filled from actual data
    
    # If revenue_7d doesn't exist, calculate it
    if 'revenue_7d' not in df.columns:
        df['revenue_7d'] = df['current_price'] * df.get('orders_7d', 0)
    
    return df


def validate_data(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate data has required columns and values"""
    
    required_cols = [
        'item_id', 'current_price', 'timestamp', 'demand_7d', 
        'rating_7d', 'time_of_day', 'season', 'day_of_week',
        'is_weekend', 'is_holiday', 'is_event'
    ]
    
    # Check for missing columns
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return False, f"Missing required columns: {missing}"
    
    # Check for null values in required columns
    null_cols = df[required_cols].isnull().sum()
    if null_cols.any():
        return False, f"Null values found in: {null_cols[null_cols > 0].to_dict()}"
    
    # Validate price is positive
    if (df['current_price'] <= 0).any():
        return False, "current_price must be positive"
    
    # Validate timestamp is positive
    if (df['timestamp'] <= 0).any():
        return False, "timestamp must be positive"
    
    return True, "Data validation passed"


def dp_preprocessing(
    data: pd.DataFrame,
    branch_id: str,
    mode: Literal["train", "predict"]
) -> dict:
    """
    Complete preprocessing pipeline for dynamic pricing
    
    Args:
        data: Raw data as DataFrame
        branch_id: Branch ID for saving/loading preprocessor
        mode: 'train' or 'predict'
    
    Returns:
        dict with transformed data
    """
    
    # Validate data
    is_valid, validation_message = validate_data(data)
    if not is_valid:
        return returnFormat("error", validation_message)
    
    print(f"✓ Valid Data: {validation_message}")
    
    # Calculate price change percentage
    data = calculate_price_change_percent(data)
    print("✓ Price change percentage calculated")
    
    # Calculate demand metrics
    data = calculate_demand_metrics(data)
    print("✓ Demand metrics calculated")
    
    if mode == "train":
        # Fit preprocessor on training data
        fit_preprocess(data)
        
        # Save preprocessor
        if branch_id:
            preprocessor_path = Path("models") / branch_id / "preprocessor.pkl"
            save_preprocessor(preprocessor_path)
    else:
        # Load existing preprocessor
        if not branch_id:
            return returnFormat("error", "Branch ID required for prediction mode")
        
        preprocessor_path = Path("models") / branch_id / "preprocessor.pkl"
        if not preprocessor_path.exists():
            return returnFormat("error", f"No preprocessor found at {preprocessor_path}")
        
        load_preprocessor(preprocessor_path)
    
    # Transform data
    transformed = transform(data)
    
    print(f"✓ Transformed data shape: {transformed.shape}")
    print("✓ Preprocessing complete!")
    
    return returnFormat("success", "Data preprocessed successfully", transformed)