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
        self.params = {
            'n_estimators': 150,          # Reduced to prevent overfitting
            'learning_rate': 0.08,        # Slightly higher for faster convergence
            'max_depth': 3,               # Shallower trees
            'min_child_weight': 5,        # Higher regularization
            'subsample': 0.9,             # More data per tree
            'colsample_bytree': 0.9,      # More features per tree
            'reg_alpha': 0.05,            # L1 regularization
            'reg_lambda': 0.1,            # L2 regularization
            'random_state': 42,
            'verbosity': 0,
            'objective': 'reg:squarederror',
            'eval_metric': 'mae'          # Focus on MAE
        }
        self.model = None
        
    def fit(self, X, y):
        """Enhanced fit with early stopping"""
        # Split for early stopping
        if len(X) > 20:
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            self.model = XGBRegressor(**self.params)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
        else:
            self.model = XGBRegressor(**self.params)
            self.model.fit(X, y)
        
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

class ModelOptimizer:
    """Optimize model hyperparameters"""
    
    def __init__(self):
        self.optimization_budget = 20  # trials per model
    
    def optimize_random_forest(self, X, y):
        """Optimize Random Forest"""
        best_params = {
            'n_estimators': 100,
            'max_depth': 4,
            'min_samples_leaf': 3,
            'min_samples_split': 5,
            'random_state': 42
        }
        
        # Quick grid search for critical parameters
        param_grid = {
            'max_depth': [3, 4, 5],
            'min_samples_leaf': [2, 3, 5],
            'n_estimators': [50, 100, 150]
        }
        
        best_score = float('inf')
        
        for max_depth in param_grid['max_depth']:
            for min_samples_leaf in param_grid['min_samples_leaf']:
                for n_estimators in param_grid['n_estimators']:
                    params = {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'min_samples_leaf': min_samples_leaf,
                        'min_samples_split': 5,
                        'random_state': 42,
                        'n_jobs': -1
                    }
                    
                    # Quick CV evaluation
                    score = self._evaluate_params(RandomForestRegressor(**params), X, y)
                    
                    if score < best_score:
                        best_score = score
                        best_params = params
        
        print(f"✅ RF optimized: MAE improved to {best_score:.4f}")
        return best_params
    
    def optimize_lightgbm(self, X, y):
        """Optimize LightGBM"""
        param_grid = {
            'n_estimators': [100, 150, 200],
            'learning_rate': [0.05, 0.08, 0.1],
            'max_depth': [3, 4, 5],
            'num_leaves': [15, 31, 50]
        }
        
        best_params = {
            'n_estimators': 100,
            'learning_rate': 0.05,
            'max_depth': 3,
            'num_leaves': 15,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbose': -1,
            'force_col_wise': True
        }
        
        best_score = float('inf')
        
        # Simplified optimization (avoid overfitting on small data)
        for lr in param_grid['learning_rate']:
            for depth in param_grid['max_depth']:
                params = best_params.copy()
                params.update({'learning_rate': lr, 'max_depth': depth})
                
                score = self._evaluate_params(LGBMRegressor(**params), X, y)
                
                if score < best_score:
                    best_score = score
                    best_params = params
        
        print(f"✅ LightGBM optimized: MAE improved to {best_score:.4f}")
        return best_params
    
    def _evaluate_params(self, model, X, y):
        """Quick parameter evaluation"""
        if len(X) < 10:
            return float('inf')
        
        # Simple holdout validation
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            return mean_absolute_error(y_test, pred)
        except:
            return float('inf')

class ForecastingEngine:
    """Main forecasting engine for IPH prediction"""
    
    def __init__(self, data_path='data/historical_data.csv', models_path='data/models/'):
        self.optimizer = ModelOptimizer()
        
        # ✅ OPTIMIZED MODELS
        self.models = {
            'KNN': KNeighborsRegressor(n_neighbors=5, weights='distance'),
            'Random_Forest': None,  # Will be optimized
            'LightGBM': None,       # Will be optimized  
            'XGBoost_Advanced': XGBoostAdvanced()
        }

    def _optimize_models_for_data(self, X, y):
        """Optimize models based on current data"""
        print("🔧 Optimizing models for current dataset...")
        
        # Optimize Random Forest
        rf_params = self.optimizer.optimize_random_forest(X, y)
        self.models['Random_Forest'] = RandomForestRegressor(**rf_params)
        
        # Optimize LightGBM
        lgb_params = self.optimizer.optimize_lightgbm(X, y)
        self.models['LightGBM'] = LGBMRegressor(**lgb_params)
        
        print("✅ Model optimization completed")

    def prepare_features_safe(self, df, split_index=None):
        """Leak-free feature preparation"""
        df_copy = df.copy()
        
        if split_index is not None:
            # Training: only use data up to split_index
            train_data = df_copy.iloc[:split_index]
            test_data = df_copy.iloc[split_index:]
            
            # Calculate features separately
            train_features = self._calculate_features(train_data)
            test_features = self._calculate_features_test(test_data, train_data)
            
            return train_features, test_features
        else:
            # Full dataset (for production forecasting only)
            return self._calculate_features(df_copy)

    def _calculate_features(self, df):
        """Calculate features without future data"""
        df = df.copy()
        
        # Lag features (safe)
        for lag in [1, 2, 3, 4]:
            df[f'Lag_{lag}'] = df['Indikator_Harga'].shift(lag)
        
        # Moving averages (safe)
        df['MA_3'] = df['Indikator_Harga'].rolling(window=3, min_periods=1).mean()
        df['MA_7'] = df['Indikator_Harga'].rolling(window=7, min_periods=1).mean()
        
        return df.dropna(subset=self.feature_cols)

    def _calculate_features_test(self, test_df, train_df):
        """Calculate test features using only historical data"""
        combined_df = pd.concat([train_df, test_df]).reset_index(drop=True)
        train_size = len(train_df)
        
        # Calculate features on combined data
        featured_df = self._calculate_features(combined_df)
        
        # Return only test portion
        return featured_df.iloc[train_size:].dropna(subset=self.feature_cols)

    def train_and_evaluate_models(self, df):
        """ENHANCED training with all fixes"""
        print("🚀 Starting ENHANCED model training with all fixes...")
        
        # Sort by date
        df = df.sort_values('Tanggal').reset_index(drop=True)
        
        if len(df) < 15:
            print("⚠️ Small dataset: Using time series CV")
            return self._train_with_time_series_cv_fixed(df)
        
        # Time-based split
        test_size = max(5, min(int(0.2 * len(df)), len(df) // 4))
        split_index = len(df) - test_size
        
        # ✅ FIXED: Leak-free feature preparation
        train_df, test_df = self.prepare_features_safe(df, split_index)
        
        X_train = train_df[self.feature_cols].values
        y_train = train_df['Indikator_Harga'].values
        X_test = test_df[self.feature_cols].values
        y_test = test_df['Indikator_Harga'].values
        
        # ✅ NEW: Feature selection
        best_features = self.select_best_features(X_train, y_train)
        X_train = X_train[:, best_features]
        X_test = X_test[:, best_features]
        
        # ✅ NEW: Model optimization
        self._optimize_models_for_data(X_train, y_train)
        
        # Train models
        results, trained_models = self._train_with_regular_split(X_train, X_test, y_train, y_test)
        
        # ✅ NEW: Create ensemble
        if len(trained_models) >= 2:
            ensemble_model, ensemble_weights = self.create_ensemble_model(trained_models, results)
            
            # Evaluate ensemble
            ensemble_pred = ensemble_model.predict(X_test)
            ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
            
            # Add ensemble to results if it's better
            best_individual_mae = min(results[name]['mae'] for name in results.keys())
            
            if ensemble_mae < best_individual_mae:
                results['Ensemble'] = {
                    'model': ensemble_model,
                    'mae': float(ensemble_mae),
                    'rmse': float(np.sqrt(mean_squared_error(y_test, ensemble_pred))),
                    'r2_score': float(r2_score(y_test, ensemble_pred)),
                    'mape': self._calculate_mape(y_test, ensemble_pred),
                    'training_time': 0.0,
                    'data_size': len(X_train),
                    'test_size': len(X_test),
                    'trained_at': datetime.now().isoformat(),
                    'feature_importance': ensemble_model.get_feature_importance(),
                    'is_ensemble': True,
                    'ensemble_weights': ensemble_weights,
                    'ensemble_models': ensemble_model.model_names
                }
                
                print(f"🎉 Ensemble improved MAE: {ensemble_mae:.4f} vs {best_individual_mae:.4f}")
        
        # Update best model flag
        best_model_name = min(results.keys(), key=lambda x: results[x]['mae'])
        for model_name in results:
            results[model_name]['is_best'] = (model_name == best_model_name)
        
        return results, trained_models
    
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

    def _train_with_time_series_cv(self, df):
        """FIXED time series cross-validation"""
        validator = TimeSeriesValidator(n_splits=3, test_size=0.25)
        splits = validator.split(df)
        
        if not splits:
            raise ValueError("Insufficient data for time series validation")
        
        results = {}
        
        for name, model in self.models.items():
            print(f"🤖 Training {name} with {len(splits)}-fold time series CV...")
            
            cv_scores = []
            
            for train_end, test_start, test_end in splits:
                # ✅ FIXED: Proper feature preparation per split
                train_subset = df.iloc[:train_end]
                test_subset = df.iloc[test_start:test_end]
                
                # Prepare features safely
                train_features = self._calculate_features(train_subset)
                test_features = self._calculate_features_test(test_subset, train_subset)
                
                if len(train_features) < 5 or len(test_features) < 1:
                    continue
                
                X_train_cv = train_features[self.feature_cols].values
                y_train_cv = train_features['Indikator_Harga'].values
                X_test_cv = test_features[self.feature_cols].values
                y_test_cv = test_features['Indikator_Harga'].values
                
                # Train and evaluate
                model_cv = self._create_fresh_model(name)
                model_cv.fit(X_train_cv, y_train_cv)
                y_pred_cv = model_cv.predict(X_test_cv)
                
                mae_cv = mean_absolute_error(y_test_cv, y_pred_cv)
                cv_scores.append(mae_cv)
            
            # Final model training on all data
            full_features = self._calculate_features(df)
            X_full = full_features[self.feature_cols].values
            y_full = full_features['Indikator_Harga'].values
            
            model.fit(X_full, y_full)
            
            # Use CV scores for evaluation
            mae = np.mean(cv_scores) if cv_scores else float('inf')
            rmse = np.sqrt(np.mean([score**2 for score in cv_scores])) if cv_scores else float('inf')
            
            results[name] = {
                'model': model,
                'mae': float(mae),
                'rmse': float(rmse),
                'r2_score': 0.0,  # Don't trust R² in CV
                'cv_scores': cv_scores,
                'cv_std': float(np.std(cv_scores)) if cv_scores else 0.0,
                'validation_method': 'time_series_cv_fixed'
            }
        
        return results, {name: result['model'] for name, result in results.items()}

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
        """Enhanced forecasting with better uncertainty estimation"""
        print(f"🔮 Generating {n_steps}-step forecast with enhanced uncertainty...")
        
        predictions = []
        uncertainties = []
        current_features = last_features.copy()
        
        # Calculate historical volatility for better confidence intervals
        historical_volatility = np.std(last_features[:4])
        
        # Monte Carlo simulation for uncertainty
        n_simulations = 50
        
        for step in range(n_steps):
            step_predictions = []
            
            # Multiple predictions with noise injection
            for sim in range(n_simulations):
                # Add small noise to features (representing uncertainty)
                noise_factor = 0.01 * historical_volatility * np.sqrt(step + 1)
                noisy_features = current_features + np.random.normal(0, noise_factor, current_features.shape)
                
                pred = model.predict(noisy_features.reshape(1, -1))[0]
                step_predictions.append(pred)
            
            # Calculate statistics
            mean_pred = np.mean(step_predictions)
            std_pred = np.std(step_predictions)
            
            predictions.append(mean_pred)
            uncertainties.append(std_pred)
            
            # ✅ IMPROVED: Better feature updating
            new_features = self._update_features_smartly(current_features, mean_pred, predictions, step)
            current_features = new_features
        
        # Calculate confidence intervals (more sophisticated)
        confidence_multiplier = 1.96  # 95% confidence
        lower_bounds = [pred - conf * confidence_multiplier for pred, conf in zip(predictions, uncertainties)]
        upper_bounds = [pred + conf * confidence_multiplier for pred, conf in zip(predictions, uncertainties)]
        
        print(f"✅ Enhanced forecast: avg={np.mean(predictions):.3f}, uncertainty={np.mean(uncertainties):.3f}")
        
        return {
            'predictions': np.array(predictions),
            'lower_bound': np.array(lower_bounds),
            'upper_bound': np.array(upper_bounds),
            'uncertainties': np.array(uncertainties),
            'confidence_width': np.mean(np.array(upper_bounds) - np.array(lower_bounds)),
            'method': 'monte_carlo_enhanced'
        }

    def _update_features(self, current_features, new_pred, all_predictions, step):
        """Smarter feature updating with trend consideration"""
        new_features = np.zeros_like(current_features)
        
        # Update lag features
        new_features[0] = new_pred  # Lag_1
        for i in range(1, min(4, len(current_features))):
            new_features[i] = current_features[i-1]  # Shift previous lags
        
        # Update moving averages with trend awareness
        if step == 0:
            # First step: use current + historical
            new_features[4] = np.mean([new_pred, current_features[0], current_features[1]])  # MA_3
            new_features[5] = np.mean(current_features[:4])  # MA_7
        else:
            # Later steps: use recent predictions with exponential weighting
            recent_preds = all_predictions[-min(3, len(all_predictions)):]
            weights = np.exp(np.arange(len(recent_preds)))  # Exponential weights
            weights = weights / weights.sum()
            
            new_features[4] = np.average(recent_preds, weights=weights)  # Weighted MA_3
            
            # MA_7 with longer history
            recent_preds_7 = all_predictions[-min(7, len(all_predictions)):]
            if len(recent_preds_7) > 1:
                new_features[5] = np.mean(recent_preds_7)
            else:
                new_features[5] = new_pred
        
        return new_features
     
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
    
    def select_best_features(self, X, y, max_features=6):
        """Automatic feature selection"""
        from sklearn.feature_selection import SelectKBest, f_regression
        from sklearn.model_selection import cross_val_score
        
        if len(X) < 20:
            return list(range(X.shape[1]))  # Use all features for small datasets
        
        # Recursive feature elimination with time series CV
        best_score = float('inf')
        best_features = list(range(X.shape[1]))
        
        for n_features in range(3, min(max_features + 1, X.shape[1] + 1)):
            selector = SelectKBest(f_regression, k=n_features)
            X_selected = selector.fit_transform(X, y)
            
            # Quick validation with simple model
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=0.1)
            
            # Time series CV
            scores = []
            for i in range(3):
                split_idx = int(0.7 * len(X_selected)) + i * int(0.1 * len(X_selected))
                if split_idx < len(X_selected) - 2:
                    X_train_fs = X_selected[:split_idx]
                    X_test_fs = X_selected[split_idx:split_idx+2]
                    y_train_fs = y[:split_idx]
                    y_test_fs = y[split_idx:split_idx+2]
                    
                    model.fit(X_train_fs, y_train_fs)
                    pred = model.predict(X_test_fs)
                    mae = mean_absolute_error(y_test_fs, pred)
                    scores.append(mae)
            
            avg_score = np.mean(scores) if scores else float('inf')
            
            if avg_score < best_score:
                best_score = avg_score
                best_features = selector.get_support(indices=True).tolist()
        
        print(f"✅ Selected {len(best_features)} best features: {best_features}")
        return best_features

class TimeSeriesValidator:
        """Proper time series cross-validation"""
        
        def __init__(self, n_splits=5, test_size=0.2):
            self.n_splits = n_splits
            self.test_size = test_size
        
        def split(self, df):
            """Walk-forward validation splits"""
            n_samples = len(df)
            test_samples = max(1, int(n_samples * self.test_size))
            
            splits = []
            
            for i in range(self.n_splits):
                # Expanding window approach
                train_end = n_samples - test_samples * (self.n_splits - i)
                test_start = train_end
                test_end = test_start + test_samples
                
                if train_end >= 10 and test_end <= n_samples:  # Minimum 10 samples for training
                    splits.append((train_end, test_start, test_end))
            
            return splits

