import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin
import pickle
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class XGBoostAdvanced(BaseEstimator, RegressorMixin):
    """Advanced XGBoost model with optimized parameters for time series forecasting"""
    
    def __init__(self):
        self.n_estimators = 200
        self.learning_rate = 0.05
        self.max_depth = 4
        self.min_child_weight = 3
        self.subsample = 0.8
        self.colsample_bytree = 0.8
        self.reg_alpha = 0.1
        self.reg_lambda = 0.1
        self.random_state = 42
        self.model = None
        
    def fit(self, X, y):
        """Fit the XGBoost model"""
        self.model = XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            verbosity=0
        )
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        if self.model:
            return self.model.feature_importances_
        return None

class ForecastingEngine:
    """Main forecasting engine for IPH prediction"""
    
    def __init__(self, data_path='data/historical_data.csv', models_path='data/models/'):
        self.data_path = data_path
        self.models_path = models_path
        self.feature_cols = ['Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'MA_3', 'MA_7']
        
        # Initialize models with optimized parameters
        self.models = {
            'KNN': KNeighborsRegressor(n_neighbors=5, weights='distance'),
            'Random_Forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=4, 
                min_samples_leaf=3, 
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'LightGBM': LGBMRegressor(
                n_estimators=100, 
                learning_rate=0.05, 
                max_depth=3, 
                num_leaves=15,
                reg_alpha=0.1, 
                reg_lambda=0.1, 
                random_state=42, 
                verbose=-1,
                force_col_wise=True
            ),
            'XGBoost_Advanced': XGBoostAdvanced()
        }
        
        # Create directories
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)
    
    def prepare_features(self, df):
        """Prepare lag and moving average features for time series"""
        print("🔧 Preparing features...")
        
        df_copy = df.copy()
        
        # Ensure proper datetime format
        if 'Tanggal' in df_copy.columns:
            df_copy['Tanggal'] = pd.to_datetime(df_copy['Tanggal'])
        
        # Sort by date
        df_copy = df_copy.sort_values('Tanggal').reset_index(drop=True)
        
        # Create lag features (previous values)
        for lag in [1, 2, 3, 4]:
            df_copy[f'Lag_{lag}'] = df_copy['Indikator_Harga'].shift(lag)
        
        # Create moving averages
        df_copy['MA_3'] = df_copy['Indikator_Harga'].rolling(window=3, min_periods=1).mean()
        df_copy['MA_7'] = df_copy['Indikator_Harga'].rolling(window=7, min_periods=1).mean()
        
        # Add time-based features
        if 'Tanggal' in df_copy.columns:
            df_copy['Month'] = df_copy['Tanggal'].dt.month
            df_copy['Quarter'] = df_copy['Tanggal'].dt.quarter
            df_copy['DayOfYear'] = df_copy['Tanggal'].dt.dayofyear
        
        # Remove rows with NaN values in required features
        df_clean = df_copy.dropna(subset=self.feature_cols)
        
        print(f"✅ Features prepared: {len(df_clean)} samples ready for training")
        return df_clean
    
    def train_and_evaluate_models(self, df):
        """Train all models and return performance metrics"""
        print("🚀 Starting model training and evaluation...")
        
        # Prepare features
        df_features = self.prepare_features(df)
        
        if len(df_features) == 0:
            raise ValueError("❌ No valid data after feature preparation")
        
        print(f"📊 Dataset prepared: {len(df_features)} samples")
        
        # Split features and target
        X = df_features[self.feature_cols].values
        y = df_features['Indikator_Harga'].values
        
        # Improved train/test split strategy
        if len(X) < 20:
            # For very small datasets, use time series cross-validation
            print("🔄 Using time series cross-validation for small dataset...")
            return self._train_with_time_series_cv(df_features, X, y)
        else:
            # For larger datasets, use traditional split
            test_size = max(5, min(int(0.2 * len(X)), len(X) // 4))  # 20% but at least 5, max 25%
            split_point = len(X) - test_size
            
            X_train, X_test = X[:split_point], X[split_point:]
            y_train, y_test = y[:split_point], y[split_point:]
            
            print(f"📈 Training set: {len(X_train)} samples")
            print(f"🧪 Test set: {len(X_test)} samples")
            
            return self._train_with_regular_split(X_train, X_test, y_train, y_test)

    def _train_with_regular_split(self, X_train, X_test, y_train, y_test):
        """Train with regular train/test split"""
        results = {}
        trained_models = {}
        
        for name, model in self.models.items():
            try:
                print(f"🤖 Training {name}...")
                start_time = datetime.now()
                
                # Train model
                model.fit(X_train, y_train)
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics with better handling
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                # Better R² calculation with variance check
                y_test_var = np.var(y_test)
                if len(y_test) > 1 and y_test_var > 1e-8:
                    r2 = r2_score(y_test, y_pred)
                    # Clamp R² to reasonable range
                    r2 = max(-10.0, min(1.0, r2))
                else:
                    print(f"   ⚠️ {name}: Low variance in test set, R² may be unreliable")
                    r2 = 0.0
                
                # Handle NaN/Inf in R²
                if np.isnan(r2) or np.isinf(r2):
                    r2 = 0.0

                # Calculate MAPE with better handling
                mask = np.abs(y_test) > 1e-8
                if mask.sum() > 0:
                    mape_values = np.abs((y_test[mask] - y_pred[mask]) / y_test[mask]) * 100
                    mape = float(np.mean(mape_values))
                    # Clamp MAPE to reasonable range
                    mape = min(1000.0, mape)
                else:
                    mape = 0.0

                # Store results
                results[name] = {
                    'model': model,
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'r2_score': float(r2),
                    'mape': float(mape),
                    'training_time': float(training_time),
                    'data_size': int(len(X_train)),
                    'test_size': int(len(X_test)),
                    'trained_at': datetime.now().isoformat(),
                    'feature_importance': self._get_feature_importance(model)
                }
                
                trained_models[name] = model
                print(f"✅ {name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}, MAPE={mape:.2f}%")
                
            except Exception as e:
                print(f"❌ Error training {name}: {str(e)}")
                continue
        
        if not results:
            raise ValueError("❌ No models were successfully trained")
        
        # Determine best model (lowest MAE)
        best_model_name = min(results.keys(), key=lambda x: results[x]['mae'])
        for model_name in results:
            results[model_name]['is_best'] = bool(model_name == best_model_name)
        
        print(f"🏆 Best model: {best_model_name} (MAE: {results[best_model_name]['mae']:.4f})")
        
        return results, trained_models

    def _train_with_time_series_cv(self, df_features, X, y):
        """Train with time series cross-validation for small datasets"""
        results = {}
        trained_models = {}
        
        # Use expanding window approach
        n_splits = max(3, min(5, len(X) - 5))  # 3-5 splits, ensure at least 5 samples for training
        
        for name, model in self.models.items():
            try:
                print(f"🤖 Training {name} with {n_splits}-fold CV...")
                start_time = datetime.now()
                
                cv_scores = []
                
                for i in range(n_splits):
                    # Expanding window: train on [0:train_end], test on [train_end:train_end+1]
                    train_end = len(X) - n_splits + i
                    test_idx = train_end
                    
                    if test_idx < len(X):
                        X_train_cv = X[:train_end]
                        X_test_cv = X[test_idx:test_idx+1]
                        y_train_cv = y[:train_end]
                        y_test_cv = y[test_idx:test_idx+1]
                        
                        if len(X_train_cv) >= 3:  # Minimum samples for training
                            # Create fresh model instance
                            if name == 'XGBoost_Advanced':
                                model_cv = XGBoostAdvanced()
                            else:
                                model_cv = type(model)(**model.get_params() if hasattr(model, 'get_params') else {})
                            
                            model_cv.fit(X_train_cv, y_train_cv)
                            y_pred_cv = model_cv.predict(X_test_cv)
                            
                            mae_cv = mean_absolute_error(y_test_cv, y_pred_cv)
                            cv_scores.append(mae_cv)
                
                # Train final model on all data
                model.fit(X, y)
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Use CV scores for evaluation
                if cv_scores:
                    mae = np.mean(cv_scores)
                    rmse = mae * 1.2  # Approximate RMSE
                    r2 = max(0.0, 1.0 - (mae / (np.std(y) + 1e-8))**2)  # Approximate R²
                    mape = min(200.0, mae / (np.mean(np.abs(y)) + 1e-8) * 100)  # Approximate MAPE
                else:
                    # Fallback: use simple metrics
                    y_pred_all = model.predict(X)
                    mae = mean_absolute_error(y, y_pred_all)
                    rmse = np.sqrt(mean_squared_error(y, y_pred_all))
                    r2 = 0.0  # Don't trust R² for same data
                    mape = 100.0
                
                results[name] = {
                    'model': model,
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'r2_score': float(r2),
                    'mape': float(mape),
                    'training_time': float(training_time),
                    'data_size': int(len(X)),
                    'test_size': int(n_splits),
                    'trained_at': datetime.now().isoformat(),
                    'feature_importance': self._get_feature_importance(model),
                    'cv_method': 'time_series_expanding'
                }
                
                trained_models[name] = model
                print(f"✅ {name}: MAE={mae:.4f} (CV), RMSE={rmse:.4f}, R²={r2:.4f}, MAPE={mape:.2f}%")
                
            except Exception as e:
                print(f"❌ Error training {name}: {str(e)}")
                continue
        
        if not results:
            raise ValueError("❌ No models were successfully trained")
        
        # Determine best model
        best_model_name = min(results.keys(), key=lambda x: results[x]['mae'])
        for model_name in results:
            results[model_name]['is_best'] = bool(model_name == best_model_name)
        
        print(f"🏆 Best model: {best_model_name} (MAE: {results[best_model_name]['mae']:.4f})")
        
        return results, trained_models

    def _get_feature_importance(self, model):
        """Get feature importance if available"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                # Clean NaN values
                cleaned_importance = []
                for val in importance:
                    if np.isnan(val) or np.isinf(val):
                        cleaned_importance.append(0.0)
                    else:
                        cleaned_importance.append(float(val))
                return cleaned_importance
            elif hasattr(model, 'get_feature_importance'):
                importance = model.get_feature_importance()
                if importance is not None:
                    # Clean NaN values
                    cleaned_importance = []
                    for val in importance:
                        if np.isnan(val) or np.isinf(val):
                            cleaned_importance.append(0.0)
                        else:
                            cleaned_importance.append(float(val))
                    return cleaned_importance
                else:
                    return None
            else:
                return None
        except Exception as e:
            print(f"⚠️ Error getting feature importance: {e}")
            return None
   
    def save_models(self, trained_models, results):
        """Save trained models to disk"""
        print("💾 Saving models to disk...")
        
        saved_models = []
        
        for name, model in trained_models.items():
            model_data = {
                'model': model,
                'performance': results[name],
                'feature_cols': self.feature_cols,
                'model_type': type(model).__name__,
                'saved_at': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            # Create safe filename
            safe_name = name.replace(' ', '_').replace('/', '_')
            filepath = os.path.join(self.models_path, f"{safe_name}.pkl")
            
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                print(f"✅ Saved {name} to {filepath}")
                saved_models.append({
                    'name': name,
                    'filepath': filepath,
                    'size_mb': os.path.getsize(filepath) / (1024 * 1024)
                })
                
            except Exception as e:
                print(f"❌ Error saving {name}: {str(e)}")
        
        print(f"💾 Successfully saved {len(saved_models)} models")
        return saved_models
    
    def load_model(self, model_name):
        """Load a specific model from disk"""
        safe_name = model_name.replace(' ', '_').replace('/', '_')
        filepath = os.path.join(self.models_path, f"{safe_name}.pkl")
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            print(f"✅ Loaded {model_name} from {filepath}")
            return model_data
            
        except FileNotFoundError:
            print(f"❌ Model file not found: {filepath}")
            return None
        except Exception as e:
            print(f"❌ Error loading {model_name}: {str(e)}")
            return None
    
    def forecast_multistep(self, model, last_features, n_steps):
        """Multi-step ahead forecasting with confidence intervals"""
        print(f"🔮 Generating {n_steps}-step forecast...")
        
        predictions = []
        predictions_lower = []
        predictions_upper = []
        current_features = last_features.copy()
        
        # Calculate historical volatility for confidence intervals
        historical_std = np.std(last_features[:4])  # Use lag values as proxy
        base_confidence = 1.96 * historical_std
        
        for step in range(n_steps):
            # Predict next value
            pred = model.predict(current_features.reshape(1, -1))[0]
            predictions.append(pred)
            
            # Calculate expanding confidence intervals
            confidence_factor = base_confidence * np.sqrt(step + 1) * 1.2  # Slight increase for uncertainty
            predictions_lower.append(pred - confidence_factor)
            predictions_upper.append(pred + confidence_factor)
            
            # Update features for next prediction
            new_features = np.zeros_like(current_features)
            
            # Update lag features
            new_features[0] = pred  # Lag_1
            for i in range(1, 4):  # Lag_2, Lag_3, Lag_4
                if i < len(current_features):
                    new_features[i] = current_features[i-1]
            
            # Update moving averages with more sophisticated calculation
            if step == 0:
                new_features[4] = np.mean([pred, current_features[0], current_features[1]])  # MA_3
                new_features[5] = np.mean(current_features[:4])  # MA_7
            else:
                # Use recent predictions for MA calculation
                recent_values = [pred] + [predictions[max(0, step-i)] for i in range(1, min(3, step+1))]
                new_features[4] = np.mean(recent_values[:3])
                
                recent_values_7 = [pred] + [predictions[max(0, step-i)] for i in range(1, min(7, step+1))]
                new_features[5] = np.mean(recent_values_7[:7])
            
            current_features = new_features
        
        print(f"✅ Forecast completed: avg={np.mean(predictions):.3f}, trend={'↗' if predictions[-1] > predictions[0] else '↘'}")
        
        return {
            'predictions': np.array(predictions),
            'lower_bound': np.array(predictions_lower),
            'upper_bound': np.array(predictions_upper),
            'confidence_width': np.mean(np.array(predictions_upper) - np.array(predictions_lower))
        }
        
    def generate_forecast(self, model_name, forecast_weeks=8):
        """Generate forecast using specified model"""
        print("=" * 100)
        print(f"🎯 FORECASTING ENGINE - GENERATE FORECAST:")
        print(f"   🤖 Requested model: '{model_name}'")
        print(f"   📊 Requested weeks: {forecast_weeks}")
        
        # Validate forecast weeks
        if not (4 <= forecast_weeks <= 12):
            raise ValueError("Forecast weeks must be between 4 and 12")
        
        # Load historical data
        if not os.path.exists(self.data_path):
            raise ValueError("❌ No historical data found. Please upload data first.")
        
        df = pd.read_csv(self.data_path)
        df['Tanggal'] = pd.to_datetime(df['Tanggal'])
        
        # Prepare features
        df_features = self.prepare_features(df)
        
        if len(df_features) == 0:
            raise ValueError("❌ No valid data for forecasting")
        
        print(f"   📋 Available models in self.models: {list(self.models.keys())}")
        
        # Load model
        model_data = self.load_model(model_name)
        if not model_data:
            print(f"   ❌ Model '{model_name}' not found, trying to use from memory...")
            # Try to use model from memory if available
            if model_name in self.models:
                print(f"   ✅ Found '{model_name}' in memory models")
                # Need to train the model first if not already trained
                try:
                    # Quick check if model is trained
                    test_features = df_features[self.feature_cols].iloc[-1:].values
                    _ = self.models[model_name].predict(test_features)
                    print(f"   ✅ Model '{model_name}' is ready to use")
                    model = self.models[model_name]
                    # Create dummy performance data
                    model_performance = {
                        'mae': 0.0,
                        'rmse': 0.0, 
                        'r2_score': 0.0,
                        'training_time': 0.0,
                        'trained_at': datetime.now().isoformat()
                    }
                except:
                    print(f"   🔄 Model '{model_name}' needs training...")
                    # Train the specific model
                    X = df_features[self.feature_cols].values
                    y = df_features['Indikator_Harga'].values
                    
                    model = self.models[model_name]
                    model.fit(X, y)
                    print(f"   ✅ Model '{model_name}' trained successfully")
                    
                    # Create basic performance metrics
                    model_performance = {
                        'mae': 0.0,
                        'rmse': 0.0,
                        'r2_score': 0.0, 
                        'training_time': 1.0,
                        'trained_at': datetime.now().isoformat()
                    }
            else:
                raise ValueError(f"❌ Model '{model_name}' not found in available models: {list(self.models.keys())}")
        else:
            print(f"   ✅ Loaded model '{model_name}' from disk")
            model = model_data['model']
            model_performance = model_data['performance']
        
        # Get last features for forecasting
        last_features = df_features[self.feature_cols].iloc[-1].values
        
        print(f"   📊 Using features from last data point: {df_features['Tanggal'].iloc[-1].strftime('%Y-%m-%d')}")
        print(f"   🔮 Generating {forecast_weeks} weeks forecast with model: '{model_name}'")
        
        # Generate forecast
        forecast_result = self.forecast_multistep(model, last_features, forecast_weeks)
        
        # Create forecast dates (weekly intervals)
        last_date = df_features['Tanggal'].max()
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=7), 
            periods=forecast_weeks, 
            freq='W'
        )
        
        # Prepare forecast dataframe with JSON-serializable data
        forecast_df = pd.DataFrame({
            'Tanggal': [date.strftime('%Y-%m-%d') for date in forecast_dates],
            'Prediksi': [float(pred) for pred in forecast_result['predictions']],
            'Batas_Bawah': [float(lower) for lower in forecast_result['lower_bound']],
            'Batas_Atas': [float(upper) for upper in forecast_result['upper_bound']],
            'Model': model_name,  # Store the actual model name used
            'Confidence_Width': float(forecast_result['confidence_width']),
            'Generated_At': datetime.now().isoformat()
        })
        
        # Add forecast metadata with JSON-serializable types
        forecast_summary = {
            'avg_prediction': float(forecast_result['predictions'].mean()),
            'trend': 'Naik' if forecast_result['predictions'][-1] > forecast_result['predictions'][0] else 'Turun',
            'volatility': float(np.std(forecast_result['predictions'])),
            'confidence_avg': float(forecast_result['confidence_width']),
            'min_prediction': float(forecast_result['predictions'].min()),
            'max_prediction': float(forecast_result['predictions'].max())
        }
        
        print(f"   ✅ Forecast generated successfully!")
        print(f"      - Model used: '{model_name}'")
        print(f"      - Forecast weeks: {forecast_weeks}")
        print(f"      - Average prediction: {forecast_summary['avg_prediction']:.3f}%")
        print(f"      - Trend: {forecast_summary['trend']}")
        print("=" * 100)
        
        return forecast_df, model_performance, forecast_summary

    def get_available_models(self):
        """Get list of available saved models"""
        if not os.path.exists(self.models_path):
            return []
        
        model_files = [f for f in os.listdir(self.models_path) if f.endswith('.pkl')]
        available_models = []
        
        for file in model_files:
            model_name = file.replace('.pkl', '').replace('_', ' ')
            filepath = os.path.join(self.models_path, file)
            
            try:
                # Get file info
                stat = os.stat(filepath)
                available_models.append({
                    'name': model_name,
                    'filename': file,
                    'size_mb': stat.st_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            except:
                continue
        
        return available_models