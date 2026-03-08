"""
Comprehensive ML Pipeline for RF Jamming Dataset - GPU Optimized Edition
Supports streaming data generators for 14.5GB datasets on Kaggle T4 GPU
Features:
  - tf.data.Dataset pipeline for memory-efficient data streaming
  - GPU acceleration for Deep Learning and XGBoost
  - Separate pipelines for Traditional ML (10K files) and Deep Learning (Full dataset)
  - Explicit GPU device detection and utilization
"""

import os
import glob
import pandas as pd
import numpy as np
import random
import warnings
import gc
from datetime import datetime
from pathlib import Path
try:
    import joblib
except ImportError:
    from sklearn.externals import joblib

# Data processing & visualization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

# Traditional ML models
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Deep Learning with GPU support
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

warnings.filterwarnings('ignore')

# ============================================================================
# 0. GPU SETUP AND CONFIGURATION
# ============================================================================

def setup_gpu():
    """Configure GPU for TensorFlow and print device information"""
    print(f"\n{'='*70}")
    print(f"GPU SETUP AND CONFIGURATION")
    print(f"{'='*70}\n")
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    print(f"[GPU Detection]")
    print(f"  Total GPUs found: {len(gpus)}")
    
    if gpus:
        for idx, gpu in enumerate(gpus):
            print(f"  GPU {idx}: {gpu}")
        
        # Enable memory growth to prevent OOM issues
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"\n  ✓ Memory growth enabled (prevents OOM crashes)")
        except:
            print(f"\n  ⚠️  Could not enable memory growth")
        
        print(f"\n  ✓ GPU is available for TensorFlow")
        gpu_available = True
    else:
        print(f"  ⚠️  NO GPU DETECTED - Training will be slow on CPU")
        gpu_available = False
    
    # Check XGBoost GPU capability
    print(f"\n[XGBoost GPU Check]")
    try:
        test_model = xgb.XGBClassifier(tree_method='hist', device='cuda')
        print(f"  ✓ XGBoost GPU support available (device='cuda')")
        xgb_gpu_available = True
    except:
        print(f"  ⚠️  XGBoost GPU not available, will use CPU (hist)")
        xgb_gpu_available = False
    
    print(f"\n{'='*70}\n")
    
    return gpu_available, xgb_gpu_available


# ============================================================================
# 1. DATA GENERATOR FOR STREAMING DATASETS
# ============================================================================

class FastKerasDataGenerator(keras.utils.Sequence):
    """
    High-speed chunking data generator for Keras.
    Reads N full files into a memory buffer to eliminate Disk I/O bottlenecks.
    """
    
    def __init__(self, csv_files, batch_size=2048, feature_cols=None, 
                 files_per_chunk=20, random_seed=42, scaler=None):
        """
        Initialize high-speed Keras data generator.
        
        Args:
            csv_files: List of CSV file paths
            batch_size: Batch size (default 2048 for GPU saturation)
            feature_cols: Feature column names
            files_per_chunk: Number of files to load into buffer at once
            random_seed: Random seed
            scaler: Fitted StandardScaler for feature normalization
        """
        self.csv_files = csv_files
        self.batch_size = batch_size
        self.feature_cols = feature_cols or [
            'freq1', 'noise', 'max_magnitude', 'total_gain_db',
            'base_pwr_db', 'rssi', 'relpwr_db', 'avgpwr_db', 'rssi_dbm', 'scan_type'
        ]
        self.files_per_chunk = files_per_chunk
        self.random_seed = random_seed
        self.scaler = scaler
        
        # Estimate dataset size
        estimated_total_rows = len(csv_files) * 1000
        self.steps_per_epoch = max(1, estimated_total_rows // batch_size)
        
        self.on_epoch_end()
    
    def _read_file_chunk(self, chunk_files):
        """Read N files at once into a single buffer"""
        dataframes = []
        
        for file_path in chunk_files:
            try:
                df = pd.read_csv(file_path)
                
                # Extract label from path
                if 'benign' in file_path.lower():
                    df['label'] = 0
                elif 'malicious' in file_path.lower():
                    df['label'] = 1
                else:
                    continue
                
                # Add scan type
                df['scan_type'] = 1 if 'active_scan' in file_path.lower() else 0
                
                # RSSI conversion
                if 'rssi' in df.columns:
                    df['rssi_dbm'] = df['rssi'] - 95
                
                # Select available columns
                available_cols = [col for col in self.feature_cols if col in df.columns]
                df_clean = df[available_cols + ['label']].dropna()
                
                if len(df_clean) > 0:
                    dataframes.append(df_clean)
            
            except:
                continue
        
        if not dataframes:
            return None, None
        
        # Combine all files in chunk
        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        X = combined_df[available_cols].values.astype(np.float32)
        y = combined_df['label'].values.astype(np.float32)
        
        # Apply scaling if scaler provided
        if self.scaler is not None:
            X = self.scaler.transform(X).astype(np.float32)
        
        return X, y
    
    def __len__(self):
        """Return number of batches per epoch"""
        return self.steps_per_epoch
    
    def __getitem__(self, idx):
        """Get batch at index"""
        # Calculate which chunk of files to load
        chunk_idx = (idx * self.batch_size) // (self.files_per_chunk * 1000)
        file_start = (chunk_idx % max(1, len(self.csv_files) // self.files_per_chunk)) * self.files_per_chunk
        file_end = min(file_start + self.files_per_chunk, len(self.csv_files))
        
        current_files = self.csv_files[file_start:file_end]
        X_chunk, y_chunk = self._read_file_chunk(current_files)
        
        if X_chunk is None:
            n_features = len(self.feature_cols)
            return np.zeros((self.batch_size, n_features), dtype=np.float32), \
                   np.zeros(self.batch_size, dtype=np.float32)
        
        # Extract batch from chunk
        batch_start = (idx * self.batch_size) % max(1, len(X_chunk))
        batch_end = batch_start + self.batch_size
        
        X_batch = X_chunk[batch_start:batch_end]
        y_batch = y_chunk[batch_start:batch_end]
        
        # Pad batch if needed
        if len(X_batch) < self.batch_size:
            pad_size = self.batch_size - len(X_batch)
            X_pad = np.zeros((pad_size, X_batch.shape[1]), dtype=np.float32)
            y_pad = np.zeros(pad_size, dtype=np.float32)
            X_batch = np.vstack([X_batch, X_pad])
            y_batch = np.concatenate([y_batch, y_pad])
        
        return X_batch, y_batch
    
    def on_epoch_end(self):
        """Shuffle file list after each epoch"""
        random.shuffle(self.csv_files)




def create_tf_dataset_from_files(csv_files, batch_size=32, feature_cols=None, 
                                 shuffle_buffer=1000, mode='train'):
    """
    Create tf.data.Dataset from CSV files for efficient streaming.
    
    Args:
        csv_files: List of CSV file paths
        batch_size: Batch size
        feature_cols: List of feature column names
        shuffle_buffer: Buffer size for shuffling
        mode: 'train' or 'eval' (for shuffle behavior)
        
    Returns:
        tf.data.Dataset optimized for GPU training
    """
    
    if feature_cols is None:
        feature_cols = ['freq1', 'noise', 'max_magnitude', 'total_gain_db',
                       'base_pwr_db', 'rssi', 'relpwr_db', 'avgpwr_db', 'rssi_dbm', 'scan_type']
    
    def load_and_preprocess_csv(file_path):
        """Load and preprocess a single CSV file"""
        try:
            df = pd.read_csv(file_path)
            
            # Extract label
            if b'benign' in file_path or 'benign' in str(file_path):
                label = 0
            elif b'malicious' in file_path or 'malicious' in str(file_path):
                label = 1
            else:
                return None, None
            
            # Add features
            df['scan_type'] = 1 if 'active_scan' in str(file_path).lower() else 0
            if 'rssi' in df.columns:
                df['rssi_dbm'] = df['rssi'] - 95
            
            # Get features
            available_cols = [col for col in feature_cols if col in df.columns]
            df_clean = df[available_cols].dropna()
            
            if len(df_clean) > 0:
                X = df_clean.values.astype(np.float32)
                y = np.full(len(X), label, dtype=np.float32)
                return X, y
            
            return None, None
            
        except Exception as e:
            return None, None
    
    def generator():
        """Generator function for tf.data.Dataset"""
        random.shuffle(csv_files)
        
        for file_path in csv_files:
            X, y = load_and_preprocess_csv(file_path)
            
            if X is not None:
                for x_sample, y_sample in zip(X, y):
                    yield x_sample, y_sample
    
    # Create dataset
    num_features = len(feature_cols)
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(num_features,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    )
    
    # Apply optimizations
    if mode == 'train':
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


# ============================================================================
# 2. TRADITIONAL ML DATA LOADER (LIMITED TO 10K FILES)
# ============================================================================

def load_traditional_ml_data(base_path, max_files=10000, random_seed=42):
    """
    Load data for traditional ML models (Random Forest, XGBoost).
    Limited to max_files to manage RAM and training time.
    
    Args:
        base_path: Path to dataset root
        max_files: Maximum number of files to load (default 10K for RAM efficiency)
        random_seed: Random seed
        
    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Loading Traditional ML Dataset")
    print(f"{'-'*70}")
    print(f"Note: Traditional ML (sklearn) limited to {max_files} files due to RAM constraints")
    
    # Find all CSV files
    all_files = glob.glob(os.path.join(base_path, '**', '*.csv'), recursive=True)
    print(f"Total CSV files found: {len(all_files)}")
    
    if len(all_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {base_path}")
    
    # Random sample
    random.seed(random_seed)
    sample_files = random.sample(all_files, min(max_files, len(all_files)))
    print(f"Using {len(sample_files)} files for Traditional ML training")
    
    dataframes = []
    
    for idx, file_path in enumerate(sample_files):
        if (idx + 1) % max(1, len(sample_files) // 10) == 0:
            print(f"  Loaded {idx + 1}/{len(sample_files)} files...")
        
        try:
            df = pd.read_csv(file_path)
            
            # Extract label
            if 'benign' in file_path.lower():
                df['label'] = 0
            elif 'malicious' in file_path.lower():
                df['label'] = 1
            else:
                continue
            
            # Add features
            df['scan_type'] = 1 if 'active_scan' in file_path.lower() else 0
            if 'rssi' in df.columns:
                df['rssi_dbm'] = df['rssi'] - 95
            
            dataframes.append(df)
            
        except Exception as e:
            continue
    
    if not dataframes:
        raise ValueError("No data loaded successfully")
    
    # Combine
    master_df = pd.concat(dataframes, ignore_index=True)
    print(f"Total rows: {len(master_df)}")
    print(f"Label distribution: {master_df['label'].value_counts().to_dict()}")
    
    # Define features
    feature_cols = ['freq1', 'noise', 'max_magnitude', 'total_gain_db', 
                    'base_pwr_db', 'rssi', 'relpwr_db', 'avgpwr_db', 'rssi_dbm', 'scan_type']
    feature_cols = [col for col in feature_cols if col in master_df.columns]
    
    # Clean data
    df_clean = master_df[feature_cols + ['label']].dropna()
    print(f"Rows after cleaning: {len(df_clean)}")
    
    X = df_clean[feature_cols].values
    y = df_clean['label'].values
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}\n")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ============================================================================
# 3. MODEL DEFINITIONS
# ============================================================================

class MLPipeline:
    """Unified ML Pipeline with GPU optimization"""
    
    def __init__(self, random_seed=42, xgb_gpu=True):
        self.random_seed = random_seed
        self.xgb_gpu = xgb_gpu
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
    
    def build_random_forest(self, n_estimators=100, max_depth=15):
        """Random Forest classifier"""
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_seed,
            n_jobs=-1,
            verbose=0
        )
        self.models['RandomForest'] = model
        return model
    
    def build_xgboost(self, n_estimators=100, max_depth=7, learning_rate=0.1):
        """XGBoost classifier with GPU optimization"""
        # Use GPU if available
        if self.xgb_gpu:
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=self.random_seed,
                tree_method='hist',
                device='cuda',
                eval_metric='logloss',
                verbosity=0
            )
            print("  [XGBoost configured for GPU (device='cuda')]")
        else:
            # Fallback to CPU with hist method (faster than default)
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=self.random_seed,
                tree_method='hist',
                eval_metric='logloss',
                verbosity=0
            )
            print("  [XGBoost configured for CPU (hist - faster than default)]")
        
        self.models['XGBoost'] = model
        return model
    
    def build_dnn(self, input_dim, hidden_units=[256, 128, 64], dropout_rate=0.3):
        """Deep Neural Network optimized for GPU"""
        model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(hidden_units[0], activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(hidden_units[1], activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(hidden_units[2], activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        self.models['DNN'] = model
        return model
    
    def build_cnn_1d(self, input_dim, num_filters=64, kernel_size=3):
        """1D CNN for spectral pattern detection (GPU optimized)"""
        model = models.Sequential([
            layers.Input(shape=(input_dim, 1)),
            
            layers.Conv1D(num_filters, kernel_size, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            
            layers.Conv1D(num_filters*2, kernel_size, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            
            layers.Conv1D(num_filters*4, kernel_size, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        self.models['CNN1D'] = model
        return model
    
    def build_lstm(self, input_dim, lstm_units=128):
        """LSTM for sequential RF data (GPU optimized)"""
        model = models.Sequential([
            layers.Input(shape=(input_dim, 1)),
            
            layers.LSTM(lstm_units, activation='relu', return_sequences=True),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.LSTM(lstm_units // 2, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        self.models['LSTM'] = model
        return model
    
    def train_traditional_models(self, X_train, y_train, X_test, y_test, verbose=True):
        """Train Random Forest and XGBoost on traditional ML subset with checkpointing"""
        print(f"\n{'='*70}")
        print(f"Training Traditional ML Models (Random Forest + XGBoost)")
        print(f"{'='*70}\n")
        
        # Random Forest with fault tolerance
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Training Random Forest...")
        try:
            rf_model = self.build_random_forest()
            rf_model.fit(X_train, y_train)
            rf_score = rf_model.score(X_test, y_test)
            print(f"  ✓ Random Forest Test Accuracy: {rf_score:.4f}")
            
            # Save checkpoint
            try:
                joblib.dump(rf_model, 'random_forest_model.joblib')
                print(f"  ✓ Random Forest model saved: random_forest_model.joblib")
            except Exception as save_err:
                print(f"  ⚠️  Could not save Random Forest model: {save_err}")
        
        except Exception as e:
            print(f"  ❌ Random Forest training failed: {e}")
            print(f"  Continuing with next model...\n")
        
        # XGBoost with fault tolerance
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Training XGBoost...")
        try:
            xgb_model = self.build_xgboost()
            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=0
            )
            xgb_score = xgb_model.score(X_test, y_test)
            print(f"  ✓ XGBoost Test Accuracy: {xgb_score:.4f}")
            
            # Save checkpoint
            try:
                joblib.dump(xgb_model, 'xgboost_model.joblib')
                print(f"  ✓ XGBoost model saved: xgboost_model.joblib")
            except Exception as save_err:
                print(f"  ⚠️  Could not save XGBoost model: {save_err}")
        
        except Exception as e:
            print(f"  ❌ XGBoost training failed: {e}")
            print(f"  Continuing with next models...\n")
    
    def train_deep_learning_models_streaming(self, base_path, csv_files, epochs=30, 
                                            batch_size=2048, validation_split=0.1, scaler=None):
        """Train Deep Learning models using optimized buffered I/O generator"""
        print(f"\n{'='*70}")
        print(f"Training Deep Learning Models (Optimized Buffered I/O - 20 files/chunk)")
        print(f"{'='*70}\n")
        
        feature_cols = ['freq1', 'noise', 'max_magnitude', 'total_gain_db', 
                       'base_pwr_db', 'rssi', 'relpwr_db', 'avgpwr_db', 'rssi_dbm', 'scan_type']
        
        # Base callbacks (EarlyStopping and ReduceLROnPlateau) - shared across models
        base_callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        ]
        
        # Create data generator
        print(f"Creating optimized data generator ({len(csv_files)} files, batch_size={batch_size})...")
        train_generator = FastKerasDataGenerator(
            csv_files, 
            batch_size=batch_size,
            feature_cols=feature_cols,
            files_per_chunk=20,
            scaler=scaler
        )
        
        input_dim = len(feature_cols)
        steps_per_epoch = len(train_generator)
        
        # Train DNN with fault tolerance and checkpointing
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Training DNN (optimized streaming)...")
        try:
            dnn_model = self.build_dnn(input_dim)
            dnn_checkpoint = ModelCheckpoint('dnn_best_weights.keras', monitor='val_loss', 
                                            save_best_only=True, verbose=0)
            dnn_callbacks = base_callbacks + [dnn_checkpoint]
            
            dnn_model.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                callbacks=dnn_callbacks,
                verbose=1
            )
            
            # Save final DNN model
            dnn_model.save('dnn_model_final.keras')
            print(f"  ✓ DNN training complete - Model saved: dnn_model_final.keras")
        
        except Exception as e:
            print(f"  ❌ DNN training failed: {e}")
            print(f"  Continuing with next model...\n")
        
        # Train CNN1D with fault tolerance and checkpointing
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Training CNN1D (optimized streaming)...")
        try:
            cnn_model = self.build_cnn_1d(input_dim)
            
            # Custom generator that reshapes for Conv1D
            class CNN1DGenerator(FastKerasDataGenerator):
                def __getitem__(self, idx):
                    X, y = super().__getitem__(idx)
                    return X.reshape(X.shape[0], X.shape[1], 1), y
            
            cnn_generator = CNN1DGenerator(
                csv_files,
                batch_size=batch_size,
                feature_cols=feature_cols,
                files_per_chunk=20,
                scaler=scaler
            )
            
            cnn_checkpoint = ModelCheckpoint('cnn_best_weights.keras', monitor='val_loss', 
                                            save_best_only=True, verbose=0)
            cnn_callbacks = base_callbacks + [cnn_checkpoint]
            
            cnn_model.fit(
                cnn_generator,
                steps_per_epoch=len(cnn_generator),
                epochs=epochs,
                callbacks=cnn_callbacks,
                verbose=1
            )
            
            # Save final CNN1D model
            cnn_model.save('cnn_model_final.keras')
            print(f"  ✓ CNN1D training complete - Model saved: cnn_model_final.keras")
        
        except Exception as e:
            print(f"  ❌ CNN1D training failed: {e}")
            print(f"  Continuing with next model...\n")
        
        # Train LSTM with fault tolerance and checkpointing
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Training LSTM (optimized streaming)...")
        try:
            lstm_model = self.build_lstm(input_dim)
            
            # Custom generator that reshapes for LSTM
            class LSTMGenerator(FastKerasDataGenerator):
                def __getitem__(self, idx):
                    X, y = super().__getitem__(idx)
                    return X.reshape(X.shape[0], X.shape[1], 1), y
            
            lstm_generator = LSTMGenerator(
                csv_files,
                batch_size=batch_size,
                feature_cols=feature_cols,
                files_per_chunk=20,
                scaler=scaler
            )
            
            lstm_checkpoint = ModelCheckpoint('lstm_best_weights.keras', monitor='val_loss', 
                                             save_best_only=True, verbose=0)
            lstm_callbacks = base_callbacks + [lstm_checkpoint]
            
            lstm_model.fit(
                lstm_generator,
                steps_per_epoch=len(lstm_generator),
                epochs=epochs,
                callbacks=lstm_callbacks,
                verbose=1
            )
            
            # Save final LSTM model
            lstm_model.save('lstm_model_final.keras')
            print(f"  ✓ LSTM training complete - Model saved: lstm_model_final.keras")
        
        except Exception as e:
            print(f"  ❌ LSTM training failed: {e}")
            print(f"  Continuing with evaluation...\n")
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all models (traditional ML + deep learning) on test set"""
        print(f"\n{'='*70}")
        print(f"Evaluating All Models on Test Set")
        print(f"{'='*70}\n")
        
        # Create 3D reshaped version of X_test for CNN1D and LSTM
        # Original X_test shape: (samples, features) -> Reshape to (samples, features, 1)
        X_test_3d = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        for model_name, model in self.models.items():
            print(f"Evaluating {model_name}...")
            
            # Get predictions based on model type
            if hasattr(model, 'predict_proba'):
                # Traditional ML models (Random Forest, XGBoost) - use 2D X_test
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            elif model_name in ['CNN1D', 'LSTM']:
                # Deep learning models that require 3D input - use 3D X_test
                y_pred_proba = model.predict(X_test_3d, verbose=0).flatten().astype(float)
            else:
                # DNN and other models - use 2D X_test
                y_pred_proba = model.predict(X_test, verbose=0).flatten().astype(float)
            
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            self.metrics[model_name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC-AUC': roc_auc
            }
            
            self.predictions[model_name] = y_pred
            
            print(f"  ✓ {model_name}: Acc={accuracy:.4f}, F1={f1:.4f}, AUC={roc_auc:.4f}")
    
    def print_comparison_table(self):
        """Print results table"""
        if not self.metrics:
            print("No evaluation results available yet.")
            return None
        
        print(f"\n{'='*90}")
        print(f"MODEL PERFORMANCE COMPARISON")
        print(f"{'='*90}\n")
        
        comparison_df = pd.DataFrame(self.metrics).T
        comparison_df = comparison_df.round(4)
        
        print(comparison_df.to_string())
        print(f"\n{'='*90}\n")
        
        print("🏆 Best Models by Metric:")
        for metric in comparison_df.columns:
            best_model = comparison_df[metric].idxmax()
            best_score = comparison_df[metric].max()
            print(f"  • {metric}: {best_model} ({best_score:.4f})")
        
        return comparison_df
    
    def plot_results(self, save_path='./ml_results_streaming.png'):
        """Plot results"""
        if not self.metrics:
            print("No metrics to plot.")
            return
        
        comparison_df = pd.DataFrame(self.metrics).T
        
        fig, axes = plt.subplots(1, 5, figsize=(18, 5))
        fig.suptitle('RF Jamming ML Model Performance (Streaming Pipeline)', 
                     fontsize=14, fontweight='bold')
        
        for idx, metric in enumerate(comparison_df.columns):
            ax = axes[idx]
            values = comparison_df[metric].sort_values(ascending=False)
            colors = plt.cm.viridis(np.linspace(0, 1, len(values)))
            
            bars = ax.barh(range(len(values)), values.values, color=colors)
            ax.set_yticks(range(len(values)))
            ax.set_yticklabels(values.index)
            ax.set_xlabel('Score')
            ax.set_title(metric, fontweight='bold')
            ax.set_xlim([0, 1])
            
            for i, val in enumerate(values.values):
                ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Results plot saved to: {save_path}\n")


# ============================================================================
# 4. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline"""
    
    print(f"\n{'='*70}")
    print(f"RF JAMMING DATASET - GPU-OPTIMIZED ML PIPELINE")
    print(f"Kaggle T4 GPU Edition with Streaming Data Generators")
    print(f"{'='*70}\n")
    
    # Step 0: GPU setup
    print(f"[Step 0/5] GPU Setup and Configuration")
    print(f"{'-'*70}")
    gpu_available, xgb_gpu = setup_gpu()
    
    # Configuration
    BASE_PATH = '/kaggle/input/datasets/daniaherzalla/radio-frequency-jamming/'
    TRADITIONAL_ML_FILES = 10000  # Max files for traditional ML
    EPOCHS = 30
    BATCH_SIZE = 2048  # Large batch size for GPU saturation with new optimized generator
    RANDOM_SEED = 42
    
    # Check if running in Kaggle or local
    if not os.path.exists(BASE_PATH):
        print(f"⚠️  Kaggle path not found. Looking for local dataset...")
        possible_paths = [
            './data',
            '../input/datasets/daniaherzalla/radio-frequency-jamming/',
            './radio-frequency-jamming'
        ]
        BASE_PATH = None
        for path in possible_paths:
            if os.path.exists(path):
                BASE_PATH = path
                break
        
        if BASE_PATH is None:
            print("❌ Dataset not found.")
            return
    
    print(f"✓ Using dataset path: {BASE_PATH}\n")
    
    # Find all CSV files
    print(f"[Step 1/5] Scanning Dataset")
    print(f"{'-'*70}")
    all_files = glob.glob(os.path.join(BASE_PATH, '**', '*.csv'), recursive=True)
    print(f"Total CSV files found: {len(all_files)}")
    print(f"Dataset configuration:")
    print(f"  • Traditional ML: {TRADITIONAL_ML_FILES} files (RAM-limited)")
    print(f"  • Deep Learning: {len(all_files)} files (streaming from disk)")
    print()
    
    # Step 1: Load traditional ML data
    print(f"[Step 2/5] Loading Traditional ML Dataset")
    print(f"{'-'*70}")
    try:
        X_train, X_test, y_train, y_test, scaler = load_traditional_ml_data(
            BASE_PATH, 
            max_files=min(TRADITIONAL_ML_FILES, len(all_files)),
            random_seed=RANDOM_SEED
        )
    except Exception as e:
        print(f"❌ Error loading traditional ML data: {e}")
        return
    
    # Step 2: Train traditional ML models
    print(f"[Step 3/5] Training Traditional ML Models")
    print(f"{'-'*70}")
    pipeline = MLPipeline(random_seed=RANDOM_SEED, xgb_gpu=xgb_gpu)
    pipeline.train_traditional_models(X_train, y_train, X_test, y_test)
    
    # Step 3: Train deep learning with streaming
    print(f"\n[Step 4/5] Training Deep Learning Models (Full Dataset Streaming)")
    print(f"{'-'*70}")
    try:
        pipeline.train_deep_learning_models_streaming(
            BASE_PATH,
            all_files,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            scaler=scaler
        )
    except Exception as e:
        print(f"⚠️  Deep learning training interrupted: {e}")
    
    # Step 4: Evaluate ALL models (traditional ML + deep learning)
    print(f"\n[Step 5/5] Evaluating All Models")
    print(f"{'-'*70}")
    pipeline.evaluate_all_models(X_test, y_test)
    
    # Final results
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS")
    print(f"{'='*70}")
    comparison_df = pipeline.print_comparison_table()
    
    # Save results
    try:
        results_path = './model_comparison_results_streaming.csv'
        if comparison_df is not None:
            comparison_df.to_csv(results_path)
            print(f"✓ Results saved to: {results_path}")
    except Exception as e:
        print(f"⚠️  Could not save results: {e}")
    
    # Visualize
    try:
        pipeline.plot_results()
    except Exception as e:
        print(f"⚠️  Could not save plot: {e}")
    
    print(f"{'='*70}")
    print(f"✓ Pipeline completed successfully!")
    print(f"{'='*70}\n")
    
    # Cleanup
    gc.collect()
    
    return pipeline, comparison_df


if __name__ == '__main__':
    pipeline, results = main()
