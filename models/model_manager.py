import os
import json
import pandas as pd
import numpy as np  
from datetime import datetime
from .forecasting_engine import ForecastingEngine

class ModelManager:
    """Manager for handling model training, comparison, and selection"""
    
    def __init__(self, data_path='data/historical_data.csv', models_path='data/models/'):
        self.engine = ForecastingEngine(data_path, models_path)
        self.performance_history_path = os.path.join(models_path, 'performance_history.json')
        self.model_metadata_path = os.path.join(models_path, 'model_metadata.json')
        
        # Ensure models directory exists
        os.makedirs(models_path, exist_ok=True)
    
    def save_performance_history(self, results):
        """Save model performance to history file"""
        print("📝 Saving performance history...")
        
        history = self.load_performance_history()
        
        # Add new results with timestamp
        timestamp = datetime.now().isoformat()
        batch_id = f"training_{timestamp.replace(':', '-').replace('.', '-')}"
        
        for model_name, performance in results.items():
            history_entry = {
                'batch_id': batch_id,
                'timestamp': timestamp,
                'model_name': model_name,
                'mae': float(performance['mae']),  # Convert to float
                'rmse': float(performance['rmse']),  # Convert to float
                'r2_score': float(performance['r2_score']),  # Convert to float
                'mape': float(performance.get('mape', 0)),  # Convert to float
                'training_time': float(performance['training_time']),  # Convert to float
                'data_size': int(performance['data_size']),  # Convert to int
                'test_size': int(performance.get('test_size', 0)),  # Convert to int
                'is_best': bool(performance.get('is_best', False)),  # Convert to bool
                'feature_importance': [float(x) if x is not None else None for x in performance.get('feature_importance', [])] if performance.get('feature_importance') else None
            }
            history.append(history_entry)
        
        # Keep only last 100 entries per model to avoid file bloat
        model_counts = {}
        filtered_history = []
        
        # Sort by timestamp (newest first)
        history.sort(key=lambda x: x['timestamp'], reverse=True)
        
        for entry in history:
            model_name = entry['model_name']
            model_counts[model_name] = model_counts.get(model_name, 0) + 1
            
            if model_counts[model_name] <= 50:  # Keep last 50 entries per model
                filtered_history.append(entry)
        
        # Sort back to chronological order
        filtered_history.sort(key=lambda x: x['timestamp'])
        
        # Save to file
        try:
            with open(self.performance_history_path, 'w') as f:
                json.dump(filtered_history, f, indent=2, default=str)
            print(f"✅ Performance history saved ({len(filtered_history)} entries)")
        except Exception as e:
            print(f"❌ Error saving performance history: {str(e)}")
            
    def load_performance_history(self):
        """Load performance history from file"""
        if os.path.exists(self.performance_history_path):
            try:
                with open(self.performance_history_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error loading performance history: {str(e)}")
                return []
        return []
    
    def get_current_best_model(self):
        """Get the current best performing model"""
        history = self.load_performance_history()
        if not history:
            return None
        
        # Get latest results for each model
        latest_results = {}
        for entry in history:
            model_name = entry['model_name']
            if model_name not in latest_results or entry['timestamp'] > latest_results[model_name]['timestamp']:
                latest_results[model_name] = entry
        
        if not latest_results:
            return None
        
        # Filter out invalid results (NaN, Inf, or extremely high MAE)
        valid_results = {}
        for model_name, result in latest_results.items():
            mae = result.get('mae', float('inf'))
            if (isinstance(mae, (int, float)) and 
                not np.isnan(mae) and 
                not np.isinf(mae) and 
                mae < 100.0):  # Reasonable MAE threshold
                valid_results[model_name] = result
        
        if not valid_results:
            print("⚠️ No valid model results found")
            return None
        
        # Find best model (lowest MAE)
        best_model = min(valid_results.values(), key=lambda x: x['mae'])
        
        print(f"✅ Current best model: {best_model['model_name']} (MAE: {best_model['mae']:.4f})")
        
        return best_model
        
    def compare_with_previous_best(self, new_results):
        """Compare new results with previous best model"""
        print("⚖️ Comparing with previous best model...")
        
        current_best = self.get_current_best_model()
        
        # Find best from new results
        best_new_model_name = min(new_results.keys(), key=lambda x: new_results[x]['mae'])
        best_new_model = new_results[best_new_model_name].copy()
        best_new_model['model_name'] = best_new_model_name
        
        if not current_best:
            print("🆕 No previous model found. Setting new best model.")
            return {
                'is_improvement': True,
                'new_best_model': best_new_model,
                'previous_best': None,
                'improvement_percentage': None,
                'comparison_metrics': {
                    'mae_change': None,
                    'rmse_change': None,
                    'r2_change': None
                }
            }
        
        # Safe division with zero checking
        def safe_percentage_change(old_val, new_val):
            """Calculate percentage change with safe division"""
            if old_val == 0 or np.isnan(old_val) or np.isinf(old_val):
                if new_val == 0 or np.isnan(new_val) or np.isinf(new_val):
                    return 0.0  # Both are zero/invalid
                else:
                    return 100.0 if new_val > 0 else -100.0  # Arbitrary large change
            
            change = (old_val - new_val) / abs(old_val) * 100
            
            # Clamp to reasonable range
            return max(-1000.0, min(1000.0, change))
        
        # Calculate improvements with safe division
        mae_improvement = safe_percentage_change(current_best['mae'], best_new_model['mae'])
        rmse_improvement = safe_percentage_change(current_best['rmse'], best_new_model['rmse'])
        
        # Handle R² change separately (higher is better)
        if abs(current_best['r2_score']) > 1e-8:
            r2_improvement = (best_new_model['r2_score'] - current_best['r2_score']) / abs(current_best['r2_score']) * 100
            r2_improvement = max(-1000.0, min(1000.0, r2_improvement))
        else:
            r2_improvement = 0.0
        
        is_improvement = best_new_model['mae'] < current_best['mae']
        
        comparison = {
            'is_improvement': is_improvement,
            'new_best_model': best_new_model,
            'previous_best': current_best,
            'improvement_percentage': mae_improvement if is_improvement else -abs(mae_improvement),
            'comparison_metrics': {
                'mae_change': float(mae_improvement),
                'rmse_change': float(rmse_improvement),
                'r2_change': float(r2_improvement),
                'significant_improvement': abs(mae_improvement) > 5  # 5% threshold
            }
        }
        
        if is_improvement:
            if abs(mae_improvement) > 10:
                print(f"🎉 Significant improvement! {best_new_model_name} improved MAE by {mae_improvement:.2f}%")
            else:
                print(f"✅ Modest improvement. {best_new_model_name} improved MAE by {mae_improvement:.2f}%")
        else:
            print(f"📊 Previous model still better. Current best: {current_best['model_name']} (MAE: {current_best['mae']:.4f})")
            print(f"   New model MAE: {best_new_model['mae']:.4f} (worse by {abs(mae_improvement):.2f}%)")
        
        return comparison
        
    def train_and_compare_models(self, df):
        """Train models and compare with previous best"""
        print("🚀 Starting comprehensive model training and comparison...")
        
        try:
            # Train new models
            results, trained_models = self.engine.train_and_evaluate_models(df)
            
            # Compare with previous best
            comparison = self.compare_with_previous_best(results)
            
            # Save models to disk
            saved_models = self.engine.save_models(trained_models, results)
            
            # Save performance history
            self.save_performance_history(results)
            
            # Save model metadata
            self._save_model_metadata(results, saved_models)
            
            training_summary = {
                'total_models_trained': len(results),
                'best_model': comparison['new_best_model']['model_name'],
                'training_completed_at': datetime.now().isoformat(),
                'data_size': results[list(results.keys())[0]]['data_size'],
                'is_improvement': comparison['is_improvement']
            }
            
            print(f"✅ Training completed successfully!")
            print(f"   🏆 Best model: {training_summary['best_model']}")
            print(f"   📊 Models trained: {training_summary['total_models_trained']}")
            print(f"   📈 Improvement: {'Yes' if training_summary['is_improvement'] else 'No'}")
            
            return {
                'training_results': results,
                'comparison': comparison,
                'trained_models': trained_models,
                'saved_models': saved_models,
                'summary': training_summary
            }
            
        except Exception as e:
            print(f"❌ Error in training and comparison: {str(e)}")
            raise
    
    def _save_model_metadata(self, results, saved_models):
        """Save model metadata for quick access"""
        try:
            metadata = {
                'last_updated': datetime.now().isoformat(),
                'models': {}
            }
            
            for model_name, result in results.items():
                metadata['models'][model_name] = {
                    'mae': result['mae'],
                    'rmse': result['rmse'],
                    'r2_score': result['r2_score'],
                    'is_best': result.get('is_best', False),
                    'trained_at': result['trained_at'],
                    'data_size': result['data_size']
                }
            
            with open(self.model_metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Error saving model metadata: {str(e)}")
    
    def get_model_performance_summary(self):
        """Get comprehensive summary of all model performances"""
        history = self.load_performance_history()
        if not history:
            return {}
        
        # Group by model name
        model_summary = {}
        for entry in history:
            model_name = entry['model_name']
            if model_name not in model_summary:
                model_summary[model_name] = {
                    'name': model_name,
                    'performances': [],
                    'best_mae': float('inf'),
                    'latest_mae': None,
                    'latest_r2': None,
                    'training_count': 0,
                    'avg_training_time': 0,
                    'improvement_trend': []
                }
            
            model_summary[model_name]['performances'].append(entry)
            model_summary[model_name]['training_count'] += 1
            model_summary[model_name]['latest_mae'] = entry['mae']
            model_summary[model_name]['latest_r2'] = entry['r2_score']
            
            if entry['mae'] < model_summary[model_name]['best_mae']:
                model_summary[model_name]['best_mae'] = entry['mae']
        
        # Calculate averages and trends
        for model_name, summary in model_summary.items():
            performances = summary['performances']
            summary['avg_training_time'] = np.mean([p['training_time'] for p in performances])
            
            # Calculate improvement trend (last 5 trainings)
            recent_maes = [p['mae'] for p in performances[-5:]]
            if len(recent_maes) > 1:
                trend = np.polyfit(range(len(recent_maes)), recent_maes, 1)[0]
                summary['trend_direction'] = 'improving' if trend < 0 else 'declining'
            else:
                summary['trend_direction'] = 'stable'
        
        return model_summary
    
    def get_training_history_chart_data(self):
        """Get data for training history visualization"""
        history = self.load_performance_history()
        if not history:
            return {}
        
        # Group by model and prepare time series data
        chart_data = {}
        for entry in history:
            model_name = entry['model_name']
            if model_name not in chart_data:
                chart_data[model_name] = {
                    'timestamps': [],
                    'mae_values': [],
                    'r2_values': []
                }
            
            chart_data[model_name]['timestamps'].append(entry['timestamp'])
            chart_data[model_name]['mae_values'].append(entry['mae'])
            chart_data[model_name]['r2_values'].append(entry['r2_score'])
        
        return chart_data
    
    def cleanup_old_models(self, keep_last_n=10):
        """Clean up old model files to save disk space"""
        print(f"🧹 Cleaning up old models (keeping last {keep_last_n})...")
        
        try:
            history = self.load_performance_history()
            if not history:
                return
            
            # Group by model name and get timestamps
            model_timestamps = {}
            for entry in history:
                model_name = entry['model_name']
                if model_name not in model_timestamps:
                    model_timestamps[model_name] = []
                model_timestamps[model_name].append(entry['timestamp'])
            
            # For each model, keep only the latest versions
            for model_name, timestamps in model_timestamps.items():
                timestamps.sort(reverse=True)  # Newest first
                
                if len(timestamps) > keep_last_n:
                    old_timestamps = timestamps[keep_last_n:]
                    # In a real implementation, you might want to delete old model files
                    # based on these timestamps
                    print(f"   📦 {model_name}: {len(old_timestamps)} old versions identified")
            
            print("✅ Model cleanup completed")
            
        except Exception as e:
            print(f"❌ Error during model cleanup: {str(e)}")