"""
RF Jamming ML Pipeline - Hybrid Architecture with Optimized I/O
Features:
  - Random Forest: In-memory training on unified data split
  - XGBoost: Out-of-core GPU training with custom DataIter
  - Deep Learning: Buffered I/O strategy (reads N files at once, yields large batches)
  - All models evaluated on unified X_test_scaled with fault tolerance
  - Optimized for Kaggle T4 GPU with 30GB RAM
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
# GLOBAL CONFIGURATION
# ============================================================================

MAX_FILES = 10000  # Maximum files to load in memory for RF and test set
BUFFER_FILES = 20  # Number of files to buffer for Keras I/O strategy
BATCH_SIZE = 2048  # Large batch size for GPU efficiency
EPOCHS = 30
RANDOM_SEED = 42

# Feature columns for RF jamming dataset
FEATURE_COLS = ['freq1', 'noise', 'max_magnitude', 'total_gain_db', 
                'base_pwr_db', 'rssi', 'relpwr_db', 'avgpwr_db', 'rssi_dbm', 'scan_type']

# Global test set and scaler (set after load_in_memory_data)
X_test_scaled_global = None
scaler_global = None
y_test_global = None

# ============================================================================
# 0. GPU SETUP AND CONFIGURATION
# ============================================================================

def setup_gpu():
    """Configure GPU for TensorFlow and validate XGBoost"""
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
        
        # Enable memory growth
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"\n  ✓ Memory growth enabled (prevents OOM)")
        except:
            print(f"\n  ⚠️  Could not enable memory growth")
        
        gpu_available = True
    else:
        print(f"  ⚠️  NO GPU DETECTED")
        gpu_available = False
    
    # Check XGBoost GPU capability
    print(f"\n[XGBoost GPU Check]")
    try:
        test_model = xgb.XGBClassifier(tree_method='hist', device='cuda')
        print(f"  ✓ XGBoost GPU support available (device='cuda')")
        xgb_gpu = True
    except:
        print(f"  ⚠️  XGBoost GPU not available, will use CPU")
        xgb_gpu = False
    
    print(f"\n{'='*70}\n")
    
    return gpu_available, xgb_gpu


# ============================================================================
# 1. UNIFIED IN-MEMORY DATA LOADING & SCALING
# ============================================================================

def load_in_memory_data(csv_files, random_seed=RANDOM_SEED):
    """
    Load up to MAX_FILES into memory, preprocess, split, and scale.
    Returns X_train_scaled, X_test_scaled, y_train, y_test, scaler, all_files
    
    This unified split is used for:
    - Training Random Forest
    - Final evaluation of ALL models (RF, XGBoost, DNN, CNN1D, LSTM)
    """
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Loading In-Memory Data")
    print(f"{'-'*70}")
    print(f"Note: Loading up to {MAX_FILES} files into memory")
    
    # Random sample
    random.seed(random_seed)
    sample_files = random.sample(csv_files, min(MAX_FILES, len(csv_files)))
    print(f"Using {len(sample_files)} files for unified data split")
    
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
            
            # Add scan type
            df['scan_type'] = 1 if 'active_scan' in file_path.lower() else 0
            
            # RSSI conversion
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
    
    # Select available features
    available_cols = [col for col in FEATURE_COLS if col in master_df.columns]
    df_clean = master_df[available_cols + ['label']].dropna()
    print(f"Rows after cleaning: {len(df_clean)}")
    
    X = df_clean[available_cols].values.astype(np.float32)
    y = df_clean['label'].values.astype(np.float32)
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    
    print(f"Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}\n")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, available_cols


# ============================================================================
# 2. CUSTOM XGBOOST DATA ITERATOR (OUT-OF-CORE GPU TRAINING)
# ============================================================================

class XGBoostDataIter(xgb.DataIter):
    """
    Custom XGBoost DataIter for out-of-core training from CSV files.
    Chunks through files, applies scaler, yields to GPU via DeviceQuantileDMatrix.
    """
    
    def __init__(self, csv_files, scaler, feature_cols, batch_size=1024, random_seed=42):
        """
        Args:
            csv_files: List of CSV file paths
            scaler: Fitted StandardScaler
            feature_cols: List of feature column names
            batch_size: Internal chunking size
            random_seed: Seed for reproducibility
        """
        self.csv_files = csv_files
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.batch_size = batch_size
        self.random_seed = random_seed
        
        self.file_index = 0
        self.current_buffer = None
        self.buffer_index = 0
        
        random.seed(random_seed)
        random.shuffle(self.csv_files)
        
        super().__init__()
    
    def _load_next_buffer(self):
        """Load next batch of files into buffer"""
        if self.file_index >= len(self.csv_files):
            return False  # End of data
        
        dataframes = []
        
        # Load BUFFER_FILES at once
        for _ in range(BUFFER_FILES):
            if self.file_index >= len(self.csv_files):
                break
            
            file_path = self.csv_files[self.file_index]
            self.file_index += 1
            
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
            
            except:
                continue
        
        if not dataframes:
            return False
        
        # Combine and preprocess
        buffer_df = pd.concat(dataframes, ignore_index=True)
        available_cols = [col for col in self.feature_cols if col in buffer_df.columns]
        buffer_df = buffer_df[available_cols + ['label']].dropna()
        
        if len(buffer_df) == 0:
            return False
        
        # Scale features
        X = buffer_df[available_cols].values.astype(np.float32)
        X_scaled = self.scaler.transform(X).astype(np.float32)
        y = buffer_df['label'].values.astype(np.float32)
        
        self.current_buffer = (X_scaled, y)
        self.buffer_index = 0
        
        return True
    
    def next(self, input_data):
        """XGBoost calls this to get next batch"""
        if self.current_buffer is None and not self._load_next_buffer():
            return False  # No more data
        
        X_scaled, y = self.current_buffer
        
        # Get batch
        end_idx = min(self.buffer_index + self.batch_size, len(X_scaled))
        X_batch = X_scaled[self.buffer_index:end_idx]
        y_batch = y[self.buffer_index:end_idx]
        
        # Feed to XGBoost
        input_data(data=X_batch, label=y_batch)
        self.buffer_index = end_idx
        
        # Load next buffer if current exhausted
        if self.buffer_index >= len(X_scaled):
            if not self._load_next_buffer():
                return False
        
        return True
    
    def reset(self):
        """Reset iterator"""
        self.file_index = 0
        self.current_buffer = None
        self.buffer_index = 0
        random.shuffle(self.csv_files)


# ============================================================================
# 3. OPTIMIZED KERAS GENERATOR WITH BUFFERING STRATEGY
# ============================================================================

class OptimizedKerasGenerator(keras.utils.Sequence):
    """
    High-speed Keras generator with buffering strategy.
    Loads BUFFER_FILES at once, scales, shuffles, yields large batches (2048).
    """
    
    def __init__(self, csv_files, scaler, feature_cols, batch_size=2048, 
                 buffer_files=20, random_seed=42):
        """
        Args:
            csv_files: List of CSV file paths
            scaler: Fitted StandardScaler
            feature_cols: List of feature column names
            batch_size: Batch size to yield (default 2048 for GPU)
            buffer_files: Number of files to buffer at once
            random_seed: Seed for reproducibility
        """
        self.csv_files = csv_files
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.batch_size = batch_size
        self.buffer_files = buffer_files
        self.random_seed = random_seed
        
        self.file_index = 0
        self.buffer = None
        self.buffer_index = 0
        
        random.seed(random_seed)
        random.shuffle(self.csv_files)
        
        # Pre-calculate length estimate
        self.steps_per_epoch = max(1, len(csv_files) // buffer_files)
    
    def __len__(self):
        """Number of batches per epoch"""
        return self.steps_per_epoch
    
    def _load_buffer(self):
        """Load next buffer of files"""
        dataframes = []
        
        for _ in range(self.buffer_files):
            if self.file_index >= len(self.csv_files):
                break
            
            file_path = self.csv_files[self.file_index]
            self.file_index += 1
            
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
            
            except:
                continue
        
        if not dataframes:
            # Fallback: load at least one more file
            if self.file_index < len(self.csv_files):
                return self._load_buffer()
            else:
                return False
        
        # Combine and preprocess
        buffer_df = pd.concat(dataframes, ignore_index=True)
        available_cols = [col for col in self.feature_cols if col in buffer_df.columns]
        buffer_df = buffer_df[available_cols + ['label']].dropna()
        
        if len(buffer_df) == 0:
            return False
        
        # Shuffle buffer
        buffer_df = buffer_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        # Scale features
        X = buffer_df[available_cols].values.astype(np.float32)
        X_scaled = self.scaler.transform(X).astype(np.float32)
        y = buffer_df['label'].values.astype(np.float32)
        
        self.buffer = (X_scaled, y)
        self.buffer_index = 0
        
        return True
    
    def __getitem__(self, idx):
        """Get batch at index"""
        # Load initial buffer if needed
        if self.buffer is None:
            if not self._load_buffer():
                # Return dummy batch
                X_dummy = np.zeros((self.batch_size, len(self.feature_cols)), dtype=np.float32)
                y_dummy = np.zeros(self.batch_size, dtype=np.float32)
                return X_dummy, y_dummy
        
        X_scaled, y = self.buffer
        
        # Get batch
        end_idx = min(self.buffer_index + self.batch_size, len(X_scaled))
        X_batch = X_scaled[self.buffer_index:end_idx]
        y_batch = y[self.buffer_index:end_idx]
        
        self.buffer_index = end_idx
        
        # Reload buffer if exhausted
        if self.buffer_index >= len(X_scaled):
            if not self._load_buffer():
                # Shuffle and reload entire dataset
                self.file_index = 0
                random.shuffle(self.csv_files)
                self._load_buffer()
        
        # Pad batch if needed
        if len(X_batch) < self.batch_size:
            pad_size = self.batch_size - len(X_batch)
            X_batch = np.vstack([X_batch, np.tile(X_batch[-1], (pad_size, 1))])
            y_batch = np.hstack([y_batch, np.tile(y_batch[-1], pad_size)])
        
        return X_batch, y_batch
    
    def on_epoch_end(self):
        """Shuffle after each epoch"""
        self.file_index = 0
        self.buffer = None
        self.buffer_index = 0
        random.shuffle(self.csv_files)


# ============================================================================
# 4. MODEL DEFINITIONS WITH FAULT TOLERANCE
# ============================================================================

class MLPipeline:
    """Unified ML Pipeline with fault tolerance"""
    
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
        return model
    
    def build_xgboost(self, n_estimators=100, max_depth=7, learning_rate=0.1):
        """XGBoost classifier with GPU optimization"""
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
            return model, True
        else:
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=self.random_seed,
                tree_method='hist',
                eval_metric='logloss',
                verbosity=0
            )
            return model, False
    
    def build_dnn(self, input_dim):
        """Deep Neural Network"""
        model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
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
        return model
    
    def build_cnn_1d(self, input_dim):
        """1D CNN"""
        model = models.Sequential([
            layers.Input(shape=(input_dim, 1)),
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            layers.Conv1D(256, 3, activation='relu', padding='same'),
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
        return model
    
    def build_lstm(self, input_dim):
        """LSTM"""
        model = models.Sequential([
            layers.Input(shape=(input_dim, 1)),
            layers.LSTM(128, activation='relu', return_sequences=True),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.LSTM(64, activation='relu'),
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
        return model
    
    def train_random_forest(self, X_train, y_train):
        """Train RF with fault tolerance"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Training Random Forest...")
        try:
            rf_model = self.build_random_forest()
            rf_model.fit(X_train, y_train)
            self.models['RandomForest'] = rf_model
            
            # Save checkpoint
            try:
                joblib.dump(rf_model, 'random_forest_model.joblib')
                print(f"  ✓ RF trained & checkpoint saved")
            except:
                print(f"  ✓ RF trained (checkpoint save failed)")
        
        except Exception as e:
            print(f"  ❌ RF training failed: {e}")
    
    def train_xgboost_streaming(self, csv_files, feature_cols):
        """Train XGBoost with custom DataIter for out-of-core streaming"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Training XGBoost (out-of-core)...")
        try:
            xgb_model, gpu_enabled = self.build_xgboost()
            
            # Create custom iterator
            data_iter = XGBoostDataIter(
                csv_files, 
                scaler_global, 
                feature_cols,
                batch_size=1024,
                random_seed=self.random_seed
            )
            
          # Pass the iterator directly to the sklearn wrapper
            xgb_model.fit(
                X=data_iter,
                verbose=False
            )
            self.models['XGBoost'] = xgb_model
            
            # Save checkpoint
            try:
                joblib.dump(xgb_model, 'xgboost_model.joblib')
                gpu_str = " (GPU)" if gpu_enabled else " (CPU)"
                print(f"  ✓ XGBoost trained{gpu_str} & checkpoint saved")
            except:
                print(f"  ✓ XGBoost trained (checkpoint save failed)")
        
        except Exception as e:
            print(f"  ❌ XGBoost training failed: {e}")
    
    def train_dnn_streaming(self, csv_files, feature_cols):
        """Train DNN with buffered I/O"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Training DNN (streaming)...")
        try:
            input_dim = len(feature_cols)
            dnn_model = self.build_dnn(input_dim)
            
            generator = OptimizedKerasGenerator(
                csv_files,
                scaler_global,
                feature_cols,
                batch_size=BATCH_SIZE,
                buffer_files=BUFFER_FILES,
                random_seed=self.random_seed
            )
            
            checkpoint = ModelCheckpoint('dnn_best_weights.keras', monitor='val_loss',
                                        save_best_only=True, verbose=0)
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
                checkpoint
            ]
            
            dnn_model.fit(
                generator,
                steps_per_epoch=len(generator),
                epochs=EPOCHS,
                callbacks=callbacks,
                verbose=1
            )
            
            dnn_model.save('dnn_model_final.keras')
            self.models['DNN'] = dnn_model
            print(f"  ✓ DNN trained & saved")
        
        except Exception as e:
            print(f"  ❌ DNN training failed: {e}")
    
    def train_cnn1d_streaming(self, csv_files, feature_cols):
        """Train CNN1D with buffered I/O"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Training CNN1D (streaming)...")
        try:
            input_dim = len(feature_cols)
            cnn_model = self.build_cnn_1d(input_dim)
            
            # Custom generator that reshapes for Conv1D
            class CNN1DGenerator(OptimizedKerasGenerator):
                def __getitem__(self, idx):
                    X, y = super().__getitem__(idx)
                    return X.reshape(X.shape[0], X.shape[1], 1), y
            
            generator = CNN1DGenerator(
                csv_files,
                scaler_global,
                feature_cols,
                batch_size=BATCH_SIZE,
                buffer_files=BUFFER_FILES,
                random_seed=self.random_seed
            )
            
            checkpoint = ModelCheckpoint('cnn_best_weights.keras', monitor='val_loss',
                                        save_best_only=True, verbose=0)
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
                checkpoint
            ]
            
            cnn_model.fit(
                generator,
                steps_per_epoch=len(generator),
                epochs=EPOCHS,
                callbacks=callbacks,
                verbose=1
            )
            
            cnn_model.save('cnn_model_final.keras')
            self.models['CNN1D'] = cnn_model
            print(f"  ✓ CNN1D trained & saved")
        
        except Exception as e:
            print(f"  ❌ CNN1D training failed: {e}")
    
    def train_lstm_streaming(self, csv_files, feature_cols):
        """Train LSTM with buffered I/O"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Training LSTM (streaming)...")
        try:
            input_dim = len(feature_cols)
            lstm_model = self.build_lstm(input_dim)
            
            # Custom generator that reshapes for LSTM
            class LSTMGenerator(OptimizedKerasGenerator):
                def __getitem__(self, idx):
                    X, y = super().__getitem__(idx)
                    return X.reshape(X.shape[0], X.shape[1], 1), y
            
            generator = LSTMGenerator(
                csv_files,
                scaler_global,
                feature_cols,
                batch_size=BATCH_SIZE,
                buffer_files=BUFFER_FILES,
                random_seed=self.random_seed
            )
            
            checkpoint = ModelCheckpoint('lstm_best_weights.keras', monitor='val_loss',
                                        save_best_only=True, verbose=0)
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
                checkpoint
            ]
            
            lstm_model.fit(
                generator,
                steps_per_epoch=len(generator),
                epochs=EPOCHS,
                callbacks=callbacks,
                verbose=1
            )
            
            lstm_model.save('lstm_model_final.keras')
            self.models['LSTM'] = lstm_model
            print(f"  ✓ LSTM trained & saved")
        
        except Exception as e:
            print(f"  ❌ LSTM training failed: {e}")
    
    def evaluate_all_models(self, X_test, y_test, feature_cols):
        """Evaluate all trained models on unified test set"""
        print(f"\n{'='*70}")
        print(f"Evaluating All Models on Unified Test Set")
        print(f"{'='*70}\n")
        
        # 3D reshape for CNN1D and LSTM
        X_test_3d = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        for model_name, model in self.models.items():
            print(f"Evaluating {model_name}...")
            
            # Get predictions
            if hasattr(model, 'predict_proba'):
                # Traditional ML
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            elif model_name in ['CNN1D', 'LSTM']:
                # Deep learning with 3D input
                y_pred_proba = model.predict(X_test_3d, verbose=0).flatten().astype(float)
            else:
                # DNN - 2D input
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
            
            print(f"  ✓ Acc={accuracy:.4f}, F1={f1:.4f}, AUC={roc_auc:.4f}")
    
    def print_comparison_table(self):
        """Print final results"""
        if not self.metrics:
            print("No evaluation results available.")
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
    
    def plot_results(self, save_path='./ml_results_hybrid.png'):
        """Plot results"""
        if not self.metrics:
            print("No metrics to plot.")
            return
        
        comparison_df = pd.DataFrame(self.metrics).T
        
        fig, axes = plt.subplots(1, 5, figsize=(18, 5))
        fig.suptitle('RF Jamming ML Models - Hybrid Optimized Pipeline', 
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
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline"""
    
    print(f"\n{'='*70}")
    print(f"RF JAMMING DATASET - HYBRID OPTIMIZED ML PIPELINE")
    print(f"In-Memory RF + Out-of-Core XGBoost + Buffered I/O Keras")
    print(f"{'='*70}\n")
    
    # Step 0: GPU setup
    print(f"[Step 0/4] GPU Setup")
    print(f"{'-'*70}")
    gpu_available, xgb_gpu = setup_gpu()
    
    # Configuration
    BASE_PATH = '/kaggle/input/datasets/daniaherzalla/radio-frequency-jamming/'
    
    # Check if dataset exists
    if not os.path.exists(BASE_PATH):
        print(f"⚠️  Kaggle path not found. Looking for local dataset...")
        possible_paths = ['./data', '../input/datasets/daniaherzalla/radio-frequency-jamming/', './radio-frequency-jamming']
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
    print(f"[Step 1/4] Scanning Dataset")
    print(f"{'-'*70}")
    all_files = glob.glob(os.path.join(BASE_PATH, '**', '*.csv'), recursive=True)
    print(f"Total CSV files found: {len(all_files)}")
    print(f"Pipeline architecture:")
    print(f"  • RF + Eval: {MAX_FILES} files (in-memory)")
    print(f"  • XGBoost: {len(all_files)} files (out-of-core streaming)")
    print(f"  • DL: {len(all_files)} files (buffered I/O, batch_size={BATCH_SIZE})")
    print()
    
    # Step 2: Load unified in-memory data
    global X_test_scaled_global, scaler_global, y_test_global
    
    print(f"[Step 2/4] Loading Unified In-Memory Data")
    print(f"{'-'*70}")
    try:
        X_train, X_test_scaled_global, y_train, y_test_global, scaler_global, feature_cols = load_in_memory_data(
            all_files, 
            random_seed=RANDOM_SEED
        )
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Step 3: Train all models
    print(f"\n[Step 3/4] Training All Models")
    print(f"{'-'*70}")
    
    pipeline = MLPipeline(random_seed=RANDOM_SEED, xgb_gpu=xgb_gpu)
    
    # Train in-memory Random Forest
    pipeline.train_random_forest(X_train, y_train)
    
    # Train out-of-core XGBoost (on subset for speed)
    xgboost_files = random.sample(all_files, min(len(all_files), 30000))
    pipeline.train_xgboost_streaming(xgboost_files, feature_cols)
    
    # Train deep learning with buffered I/O
    dl_files = random.sample(all_files, min(len(all_files), 30000))
    pipeline.train_dnn_streaming(dl_files, feature_cols)
    pipeline.train_cnn1d_streaming(dl_files, feature_cols)
    pipeline.train_lstm_streaming(dl_files, feature_cols)
    
    # Step 4: Evaluate
    print(f"\n[Step 4/4] Evaluation")
    print(f"{'-'*70}")
    pipeline.evaluate_all_models(X_test_scaled_global, y_test_global, feature_cols)
    
    # Results
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS")
    print(f"{'='*70}")
    comparison_df = pipeline.print_comparison_table()
    
    # Save results
    try:
        comparison_df.to_csv('./model_comparison_results_hybrid.csv')
        print(f"✓ Results saved to: ./model_comparison_results_hybrid.csv")
    except:
        pass
    
    # Plot results
    try:
        pipeline.plot_results()
    except Exception as e:
        print(f"⚠️  Could not save plot: {e}")
    
    print(f"{'='*70}")
    print(f"✓ Pipeline completed successfully!")
    print(f"{'='*70}\n")
    
    gc.collect()
    
    return pipeline, comparison_df


if __name__ == '__main__':
    pipeline, results = main()
