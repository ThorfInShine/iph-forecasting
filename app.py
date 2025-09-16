from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import pandas as pd
import json
import os
import re  # TAMBAH IMPORT INI
from datetime import datetime, timedelta  # 
import plotly
import plotly.graph_objs as go
from werkzeug.utils import secure_filename
import numpy as np
import pytz
from services.visualization_service import VisualizationService
from services.forecast_service import ForecastService
from services.commodity_insight_service import CommodityInsightService


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None  # Convert NaN/Inf to null
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def clean_for_json(obj):
    """Recursively clean object for JSON serialization"""
    if isinstance(obj, dict):
        return {key: clean_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif pd.isna(obj):  # Handle pandas NA values
        return None
    else:
        return obj


app = Flask(__name__)
app.config.from_object('config.Config')

app.json_encoder = CustomJSONEncoder

# Initialize service
forecast_service = ForecastService()
visualization_service = VisualizationService(forecast_service.data_handler)
commodity_service = CommodityInsightService()

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def dashboard():
    """Main dashboard"""
    try:
        dashboard_data = forecast_service.get_dashboard_data()
        
        if dashboard_data['success']:
            return render_template('dashboard.html', 
                                 data=dashboard_data,
                                 page_title="IPH Forecasting Dashboard")
        else:
            # If there's an error getting dashboard data, show empty dashboard
            flash(f"Dashboard loading issue: {dashboard_data.get('error', 'Unknown error')}", 'warning')
            empty_data = {
                'success': False,
                'data_summary': {'total_records': 0},
                'system_status': {
                    'status_message': 'Please upload data to get started',
                    'status_level': 'warning',
                    'ready_for_use': False
                }
            }
            return render_template('dashboard.html', 
                                 data=empty_data,
                                 page_title="IPH Forecasting Dashboard")
            
    except Exception as e:
        # Handle any unexpected errors
        flash(f"Dashboard error: {str(e)}", 'error')
        empty_data = {
            'success': False,
            'data_summary': {'total_records': 0},
            'system_status': {
                'status_message': 'System error occurred. Please try uploading data.',
                'status_level': 'danger',
                'ready_for_use': False
            }
        }
        return render_template('dashboard.html', 
                             data=empty_data,
                             page_title="IPH Forecasting Dashboard")

@app.route('/upload')
def upload_page():
    """Upload page"""
    return render_template('upload.html', page_title="Upload Data")

@app.route('/api/upload-data', methods=['POST'])
def upload_data():
    """Upload new data and trigger forecasting pipeline"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'})
        
        if file and file.filename.lower().endswith(('.csv', '.xlsx')):
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read data with better error handling
            try:
                if filename.lower().endswith('.csv'):
                    # Try different encodings
                    for encoding in ['utf-8', 'latin-1', 'cp1252']:
                        try:
                            df = pd.read_csv(filepath, encoding=encoding)
                            print(f"📊 CSV loaded with {encoding} encoding")
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        return jsonify({'success': False, 'message': 'Unable to read CSV file. Please check file encoding.'})
                else:
                    df = pd.read_excel(filepath)
                    
                print(f"📊 Loaded {len(df)} rows, {len(df.columns)} columns")
                print(f"📋 Columns: {list(df.columns)}")
                
            except Exception as e:
                return jsonify({'success': False, 'message': f'Error reading file: {str(e)}'})
            
            # Validate data is not empty
            if df.empty:
                return jsonify({'success': False, 'message': 'Uploaded file is empty'})
            
            # Check minimum data requirements
            if len(df) < 3:
                return jsonify({
                    'success': False, 
                    'message': f'Insufficient data: {len(df)} rows found. Minimum 3 rows required for processing.'
                })
            
            # Get forecast weeks from form
            forecast_weeks = int(request.form.get('forecast_weeks', 8))
            if not (4 <= forecast_weeks <= 12):
                forecast_weeks = 8
            
            # Process through complete pipeline with error handling
            try:
                result = forecast_service.process_new_data_and_forecast(df, forecast_weeks)
            except ValueError as ve:
                return jsonify({
                    'success': False, 
                    'message': f'Data validation error: {str(ve)}',
                    'error_type': 'validation_error'
                })
            except Exception as pe:
                return jsonify({
                    'success': False, 
                    'message': f'Processing error: {str(pe)}',
                    'error_type': 'processing_error'
                })
            
            # Clean up uploaded file
            try:
                os.remove(filepath)
            except:
                pass
            
            # Clean result for JSON serialization
            cleaned_result = clean_for_json(result)
            
            return jsonify(cleaned_result)
        
        return jsonify({'success': False, 'message': 'Invalid file format. Please upload CSV or Excel file.'})
        
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        error_response = clean_for_json({
            'success': False, 
            'message': f'Unexpected error: {str(e)}',
            'error_type': type(e).__name__
        })
        return jsonify(error_response)

@app.route('/api/generate-forecast', methods=['POST'])
def generate_forecast():
    """Generate forecast with specified parameters"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'})
        
        model_name = data.get('model_name')  # Bisa None untuk best model
        forecast_weeks = int(data.get('forecast_weeks', 8))
        
        print("=" * 60)
        print("🔮 GENERATE FORECAST DEBUG:")
        print(f"   📊 Requested weeks: {forecast_weeks}")
        print(f"   🤖 Requested model: {model_name}")
        print(f"   📋 Raw request data: {data}")
        print("=" * 60)
        
        if not (4 <= forecast_weeks <= 12):
            return jsonify({
                'success': False, 
                'message': 'Forecast weeks must be between 4 and 12'
            })
        
        # Generate forecast with specified model and weeks
        result = forecast_service.get_current_forecast(model_name, forecast_weeks)
        
        print("=" * 60)
        print("📊 FORECAST RESULT DEBUG:")
        print(f"   ✅ Success: {result.get('success')}")
        if result.get('success'):
            forecast_data = result.get('forecast', {})
            print(f"   🤖 Model used: {forecast_data.get('model_name')}")
            print(f"   📊 Weeks generated: {forecast_data.get('weeks_forecasted')}")
            print(f"   📈 Data points: {len(forecast_data.get('data', []))}")
            print(f"   💾 Forecast saved in service memory: YES")
        else:
            print(f"   ❌ Error: {result.get('error')}")
        print("=" * 60)
        
        # Clean result for JSON
        cleaned_result = clean_for_json(result)
        
        # Force refresh dashboard data after forecast
        if cleaned_result.get('success'):
            # Trigger chart refresh by updating a timestamp
            cleaned_result['chart_refresh_token'] = datetime.now().isoformat()
            print(f"🔄 Chart refresh token: {cleaned_result['chart_refresh_token']}")
        
        return jsonify(cleaned_result)
        
    except Exception as e:
        print(f"❌ Generate forecast error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        error_response = clean_for_json({
            'success': False, 
            'message': f'Error generating forecast: {str(e)}'
        })
        return jsonify(error_response)

@app.route('/api/retrain-models', methods=['POST'])
def retrain_models():
    """Retrain models with existing data"""
    try:
        result = forecast_service.retrain_models_only()
        
        # Clean result for JSON
        cleaned_result = clean_for_json(result)
        
        return jsonify(cleaned_result)
        
    except Exception as e:
        error_response = clean_for_json({
            'success': False, 
            'message': f'Error retraining models: {str(e)}'
        })
        return jsonify(error_response)

@app.route('/api/forecast-chart')
def forecast_chart():
    """Generate forecast chart with proper data structure"""
    try:
        print("📊 FORECAST CHART - DEBUGGING:")
        
        # Get dashboard data
        dashboard_data = forecast_service.get_dashboard_data()
        
        if not dashboard_data.get('success'):
            return jsonify({
                'error': 'Dashboard data unavailable',
                'data': [],
                'layout': {}
            })
        
        current_forecast = dashboard_data.get('current_forecast')
        if not current_forecast or not current_forecast.get('data'):
            return jsonify({
                'error': 'No forecast data available',
                'data': [],
                'layout': {}
            })
        
        # Load historical data
        df = forecast_service.data_handler.load_historical_data()
        if df.empty:
            return jsonify({
                'error': 'No historical data available',
                'data': [],
                'layout': {}
            })
        
        # Clean and prepare historical data
        df = df.dropna(subset=['Tanggal', 'Indikator_Harga'])
        df['Tanggal'] = pd.to_datetime(df['Tanggal'])
        df = df.sort_values('Tanggal').reset_index(drop=True)
        
        # Get recent data for better visualization
        recent_df = df.tail(60) if len(df) > 60 else df
        
        # Prepare historical data
        hist_dates = [d.strftime('%Y-%m-%d') for d in recent_df['Tanggal']]
        hist_values = [float(v) if not pd.isna(v) else 0 for v in recent_df['Indikator_Harga']]
        
        # Prepare forecast data
        forecast_data = current_forecast['data']
        forecast_dates = []
        forecast_values = []
        lower_bounds = []
        upper_bounds = []
        
        for item in forecast_data:
            try:
                # Extract date
                date_str = item['Tanggal']
                if isinstance(date_str, str) and len(date_str) >= 10:
                    forecast_dates.append(date_str[:10])
                else:
                    forecast_dates.append(str(date_str))
                
                # Extract values with validation
                pred_val = float(item['Prediksi'])
                lower_val = float(item['Batas_Bawah'])
                upper_val = float(item['Batas_Atas'])
                
                # Skip invalid values
                if any(np.isnan(x) or np.isinf(x) for x in [pred_val, lower_val, upper_val]):
                    continue
                
                forecast_values.append(pred_val)
                lower_bounds.append(lower_val)
                upper_bounds.append(upper_val)
                
            except (KeyError, ValueError, TypeError) as e:
                print(f"Skipping invalid forecast item: {e}")
                continue
        
        if not forecast_dates or not forecast_values:
            return jsonify({
                'error': 'No valid forecast data after processing',
                'data': [],
                'layout': {}
            })
        
        print(f"   ✅ Data prepared: {len(hist_values)} historical, {len(forecast_values)} forecast")
        
        # Create chart data structure
        chart_data = []
        
        # Historical data trace
        chart_data.append({
            'x': hist_dates,
            'y': hist_values,
            'type': 'scatter',
            'mode': 'lines',
            'name': '📊 Data Historis',
            'line': {
                'color': 'rgba(74, 144, 226, 1)',
                'width': 3
            },
            'hovertemplate': '<b>%{x}</b><br>IPH: %{y:.3f}%<extra></extra>'
        })
        
        # Forecast data trace
        chart_data.append({
            'x': forecast_dates,
            'y': forecast_values,
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': '🔮 Prediksi',
            'line': {
                'color': 'rgba(255, 107, 107, 1)',
                'width': 4,
                'dash': 'dot'
            },
            'marker': {
                'size': 8,
                'symbol': 'diamond',
                'color': 'rgba(255, 107, 107, 1)',
                'line': {'color': 'white', 'width': 2}
            },
            'hovertemplate': '<b>%{x}</b><br>Prediksi: %{y:.3f}%<extra></extra>'
        })
        
        # Confidence interval trace
        if len(forecast_dates) > 0:
            # Create confidence band
            x_coords = forecast_dates + forecast_dates[::-1]
            y_coords = upper_bounds + lower_bounds[::-1]
            
            chart_data.append({
                'x': x_coords,
                'y': y_coords,
                'type': 'scatter',
                'mode': 'lines',
                'fill': 'toself',
                'fillcolor': 'rgba(255, 107, 107, 0.2)',
                'line': {'color': 'rgba(255,255,255,0)'},
                'name': '📈 Interval Kepercayaan',
                'showlegend': True,
                'hoverinfo': 'skip'
            })
        
        # Chart layout
        chart_layout = {
            'title': {
                'text': '📊 IPH Forecast & Historical Data',
                'x': 0.5,
                'font': {'size': 16, 'color': '#2D3748'}
            },
            'xaxis': {
                'title': 'Tanggal',
                'showgrid': True,
                'gridwidth': 1,
                'gridcolor': 'rgba(128,128,128,0.2)'
            },
            'yaxis': {
                'title': 'IPH (%)',
                'showgrid': True,
                'gridwidth': 1,
                'gridcolor': 'rgba(128,128,128,0.2)',
                'zeroline': True,
                'zerolinewidth': 2,
                'zerolinecolor': 'rgba(128,128,128,0.5)'
            },
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'hovermode': 'x unified',
            'showlegend': True,
            'height': 500,
            'margin': {'l': 60, 'r': 60, 't': 60, 'b': 60}
        }
        
        result = {
            'data': chart_data,
            'layout': chart_layout,
            'config': {
                'responsive': True,
                'displayModeBar': True,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d', 'autoScale2d']
            },
            '_timestamp': datetime.now().isoformat()
        }
        
        print(f"   ✅ Chart data prepared successfully")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Chart error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'error': f'Chart generation failed: {str(e)}',
            'data': [],
            'layout': {},
            'details': 'Check server console for full error details'
        })
    
@app.route('/api/model-comparison-chart')
def model_comparison_chart():
    """Generate model performance metrics with optimization"""
    try:
        # Quick response for better UX
        dashboard_data = forecast_service.get_dashboard_data()
        
        if not dashboard_data['success'] or not dashboard_data.get('model_summary'):
            return jsonify({'error': 'No model performance data available'})
        
        model_summary = dashboard_data['model_summary']
        
        # Process model metrics efficiently
        model_metrics = []
        
        # Pre-calculate min/max for normalization
        all_maes = [data.get('latest_mae', 0) for data in model_summary.values() if data.get('latest_mae', 0) > 0]
        all_times = [data.get('avg_training_time', 0) for data in model_summary.values()]
        all_counts = [data.get('training_count', 0) for data in model_summary.values()]
        
        max_mae = max(all_maes) if all_maes else 1
        min_mae = min(all_maes) if all_maes else 0
        max_time = max(all_times) if all_times else 1
        max_count = max(all_counts) if all_counts else 1
        
        for model_name, data in model_summary.items():
            mae = data.get('latest_mae', 0)
            training_time = data.get('avg_training_time', 0)
            training_count = data.get('training_count', 0)
            
            # Quick normalization
            performance_score = max(0, min(100, ((max_mae - mae) / (max_mae - min_mae + 0.001)) * 100)) if mae > 0 else 50
            speed_score = max(0, min(100, ((max_time - training_time) / (max_time + 0.001)) * 100))
            experience_score = min(100, (training_count / max(max_count, 1)) * 100)
            
            overall_score = (performance_score * 0.6 + speed_score * 0.25 + experience_score * 0.15)
            
            # Quick status determination
            if overall_score >= 80:
                status, status_color, status_icon = 'excellent', 'success', 'fa-star'
            elif overall_score >= 60:
                status, status_color, status_icon = 'good', 'info', 'fa-thumbs-up'
            elif overall_score >= 40:
                status, status_color, status_icon = 'fair', 'warning', 'fa-minus-circle'
            else:
                status, status_color, status_icon = 'poor', 'danger', 'fa-times-circle'
            
            model_metrics.append({
                'name': model_name.replace('_', ' '),
                'mae': float(mae),
                'training_time': float(training_time),
                'training_count': int(training_count),
                'performance_score': round(performance_score, 1),
                'speed_score': round(speed_score, 1),
                'experience_score': round(experience_score, 1),
                'overall_score': round(overall_score, 1),
                'status': status,
                'status_color': status_color,
                'status_icon': status_icon,
                'trend': data.get('trend_direction', 'stable')
            })
        
        # Sort by overall score
        model_metrics.sort(key=lambda x: x['overall_score'], reverse=True)
        
        result = {
            'success': True,
            'metrics': model_metrics,
            'best_model': model_metrics[0] if model_metrics else None,
            'total_models': len(model_metrics),
            'cached_at': datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error generating metrics: {str(e)}'
        })
            
@app.route('/api/data-summary')
def api_data_summary():
    """API endpoint for data summary"""
    try:
        summary = forecast_service.data_handler.get_data_summary()
        
        # Clean summary for JSON
        cleaned_summary = clean_for_json(summary)
        
        return jsonify(cleaned_summary)
    except Exception as e:
        error_response = clean_for_json({
            'error': f'Error getting data summary: {str(e)}'
        })
        return jsonify(error_response)

@app.route('/api/initialize-csv', methods=['POST'])
def initialize_csv():
    """Initialize system with CSV file (for first time setup)"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'})
        
        if file and file.filename.lower().endswith(('.csv')):
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Initialize system
            result = forecast_service.initialize_from_csv(filepath)
            
            # Clean up uploaded file
            try:
                os.remove(filepath)
            except:
                pass
            
            return jsonify(result)
        
        return jsonify({'success': False, 'message': 'Please upload a CSV file'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error initializing: {str(e)}'})

@app.route('/api/model-performance')
def get_model_performance():
    """Get model performance data for modal"""
    try:
        model_summary = forecast_service.model_manager.get_model_performance_summary()
        
        if not model_summary:
            return jsonify({
                'success': False,
                'message': 'No model performance data available'
            })
        
        # Clean data for JSON
        cleaned_models = {}
        for model_name, summary in model_summary.items():
            cleaned_models[model_name] = {
                'latest_mae': float(summary['latest_mae']) if summary['latest_mae'] is not None else 0.0,
                'best_mae': float(summary['best_mae']) if summary['best_mae'] != float('inf') else 0.0,
                'training_count': int(summary['training_count']),
                'avg_training_time': float(summary['avg_training_time']),
                'trend_direction': str(summary['trend_direction'])
            }
        
        return jsonify({
            'success': True,
            'models': cleaned_models
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting model performance: {str(e)}'
        })

@app.context_processor
def inject_datetime():
    """Inject datetime functions into templates"""
    return {
        'datetime': datetime,
        'now': datetime.now()
    }

@app.route('/api/export-data', methods=['GET'])
def export_data():
    """Export current data to CSV"""
    try:
        # Get data type from query parameter
        data_type = request.args.get('type', 'historical')  # historical, forecast, or all
        
        if data_type == 'historical':
            # Export historical data
            df = forecast_service.data_handler.load_historical_data()
            if df.empty:
                return jsonify({'success': False, 'message': 'No historical data available'})
            
            filename = f"historical_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
        elif data_type == 'forecast':
            # Export latest forecast data
            dashboard_data = forecast_service.get_dashboard_data()
            if not dashboard_data['success'] or not dashboard_data.get('current_forecast'):
                return jsonify({'success': False, 'message': 'No forecast data available'})
            
            # Convert forecast data to DataFrame
            forecast_data = dashboard_data['current_forecast']['data']
            df = pd.DataFrame(forecast_data)
            filename = f"forecast_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
        elif data_type == 'all':
            # Export combined data
            historical_df = forecast_service.data_handler.load_historical_data()
            dashboard_data = forecast_service.get_dashboard_data()
            
            if historical_df.empty:
                return jsonify({'success': False, 'message': 'No data available for export'})
            
            # Create combined export with multiple sheets info in CSV
            df = historical_df.copy()
            filename = f"complete_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            # Add forecast data if available
            if dashboard_data['success'] and dashboard_data.get('current_forecast'):
                forecast_data = dashboard_data['current_forecast']['data']
                forecast_df = pd.DataFrame(forecast_data)
                
                # Add forecast info to historical data
                df['Data_Type'] = 'Historical'
                forecast_df['Data_Type'] = 'Forecast'
                forecast_df['Indikator_Harga'] = forecast_df['Prediksi']
                
                # Combine dataframes
                df = pd.concat([df, forecast_df[['Tanggal', 'Indikator_Harga', 'Data_Type']]], ignore_index=True)
        
        else:
            return jsonify({'success': False, 'message': 'Invalid data type specified'})
        
        # Save to temporary file
        export_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        df.to_csv(export_path, index=False)
        
        # Return file info
        file_size = os.path.getsize(export_path)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': export_path,
            'file_size': file_size,
            'records': len(df),
            'download_url': f'/download/{filename}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False, 
            'message': f'Export failed: {str(e)}'
        })

@app.route('/download/<filename>')
def download_file(filename):
    """Download exported file"""
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        from flask import send_file
        return send_file(file_path, as_attachment=True, download_name=filename)
        
    except Exception as e:
        return jsonify({'error': f'Download failed: {str(e)}'}), 500
    
@app.route('/data-control')
def data_control():
    """Data Control page - menggantikan upload page"""
    try:
        # Get current data summary
        data_summary = forecast_service.data_handler.get_data_summary()
        
        # Get historical data for display
        df = forecast_service.data_handler.load_historical_data()
        
        # Convert to records for template
        historical_records = []
        if not df.empty:
            # Sort by date descending (newest first)
            df_sorted = df.sort_values('Tanggal', ascending=False)
            historical_records = df_sorted[['Tanggal', 'Indikator_Harga']].to_dict('records')
        
        return render_template('data_control.html', 
                             data_summary=data_summary,
                             historical_records=historical_records,
                             page_title="Data Control")
    except Exception as e:
        flash(f"Error loading data control: {str(e)}", 'error')
        return render_template('data_control.html', 
                             data_summary={'total_records': 0},
                             historical_records=[],
                             page_title="Data Control")

@app.route('/api/add-single-record', methods=['POST'])
def add_single_record():
    """Add single IPH record to database"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'})
        
        tanggal = data.get('tanggal')
        indikator_harga = data.get('indikator_harga')
        
        if not tanggal or indikator_harga is None:
            return jsonify({'success': False, 'message': 'Date and IPH value are required'})
        
        # Validate date format
        try:
            pd.to_datetime(tanggal)
        except:
            return jsonify({'success': False, 'message': 'Invalid date format'})
        
        # Validate IPH value
        try:
            float(indikator_harga)
        except:
            return jsonify({'success': False, 'message': 'Invalid IPH value'})
        
        # Create DataFrame with single record
        new_record = pd.DataFrame({
            'Tanggal': [tanggal],
            'Indikator_Harga': [float(indikator_harga)]
        })
        
        # Add to existing data
        result = forecast_service.data_handler.merge_and_save_data(new_record)
        
        if result:
            return jsonify({
                'success': True,
                'message': 'Record added successfully',
                'total_records': result[1]['total_records']
            })
        else:
            return jsonify({'success': False, 'message': 'Failed to add record'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error adding record: {str(e)}'})

@app.route('/visualization')
def visualization():
    """Data Visualization page"""
    return render_template('visualization.html', page_title="Visualisasi Data IPH")

@app.route('/alerts')
def alerts():
    """Alert System page"""
    return render_template('alerts.html', page_title="Sistem Peringatan IPH")

# API Routes for Visualization
@app.route('/api/visualization/decomposition')
def api_decomposition():
    """API for time series decomposition with proper JSON handling"""
    try:
        timeframe = request.args.get('timeframe', '6M')
        result = visualization_service.perform_decomposition(timeframe)
        
        if result['success']:
            # Ensure proper JSON serialization
            chart_data = result['chart']
            
            # Convert Plotly figure to JSON-serializable format
            if hasattr(chart_data, 'to_dict'):
                chart_dict = chart_data.to_dict()
            elif isinstance(chart_data, dict):
                chart_dict = chart_data
            else:
                # If it's already a JSON string, parse and re-serialize
                import json
                if isinstance(chart_data, str):
                    chart_dict = json.loads(chart_data)
                else:
                    chart_dict = chart_data
            
            # Clean the data
            cleaned_chart = clean_for_json(chart_dict)
            
            return jsonify({
                'success': True,
                'chart': cleaned_chart
            })
        else:
            return jsonify(result)
            
    except Exception as e:
        import traceback
        print(f"Decomposition API error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False, 
            'message': f'Visualization error: {str(e)}'
        })
    
@app.route('/api/visualization/moving-averages')
def api_moving_averages():
    """API for moving averages analysis with proper JSON handling"""
    try:
        timeframe = request.args.get('timeframe', '6M')
        result = visualization_service.calculate_moving_averages(timeframe)
        
        if not result['success']:
            return jsonify(result)
        
        # Result sudah dalam format dict yang benar, langsung return
        return jsonify(result)
        
    except Exception as e:
        print(f"Moving averages API error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False, 
            'message': f'Moving averages error: {str(e)}'
        })
    
@app.route('/api/visualization/volatility')
def api_volatility():
    """API for volatility analysis"""
    try:
        timeframe = request.args.get('timeframe', '6M')
        result = visualization_service.analyze_volatility(timeframe)
        
        if result['success']:
            import json
            result['chart'] = json.loads(result['chart'])
            
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/visualization/distribution')
def api_distribution():
    """API for price change distribution"""
    try:
        timeframe = request.args.get('timeframe', '6M')
        result = visualization_service.analyze_distribution(timeframe)
        
        if result['success']:
            import json
            result['pieChart'] = json.loads(result['pieChart'])
            result['histogram'] = json.loads(result['histogram'])
            
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/visualization/aggregation')
def api_aggregation():
    """API for monthly/quarterly aggregation"""
    try:
        agg_type = request.args.get('type', 'monthly')
        timeframe = request.args.get('timeframe', '6M')
        result = visualization_service.analyze_aggregation(agg_type, timeframe)
        
        if result['success']:
            import json
            result['chart'] = json.loads(result['chart'])
            
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# Alert System APIs
@app.route('/api/alerts/rules', methods=['GET', 'POST'])
def api_alert_rules():
    """API for managing alert rules"""
    if request.method == 'GET':
        # Get all alert rules
        # This would typically come from a database
        rules = [
            {
                'id': 1,
                'name': 'Threshold Alert',
                'type': 'threshold',
                'enabled': True,
                'config': {'upper': 3, 'lower': -3},
                'last_triggered': '2024-01-20T10:30:00'
            },
            {
                'id': 2,
                'name': 'Trend Reversal',
                'type': 'trend',
                'enabled': True,
                'config': {},
                'last_triggered': '2024-01-19T15:45:00'
            }
        ]
        
        statistics = {
            'critical': 2,
            'warning': 5,
            'info': 8,
            'activeRules': 5,
            'newAlerts': 3
        }
        
        return jsonify({
            'success': True,
            'rules': rules,
            'statistics': statistics
        })
    
    else:  # POST
        # Create new alert rule
        data = request.get_json()
        
        # Here you would save to database
        # For now, just return success
        return jsonify({
            'success': True,
            'message': 'Alert rule created successfully',
            'rule_id': 123
        })

@app.route('/api/alerts/recent')
def api_recent_alerts():
    """API for getting recent alerts"""
    # This would typically come from a database
    alerts = [
        {
            'id': 1,
            'type': 'threshold',
            'priority': 'critical',
            'title': 'IPH Melampaui Batas Kritis',
            'message': 'IPH mencapai 4.2%, melampaui batas kritis 3%',
            'timestamp': '2024-01-20T14:30:00',
            'acknowledged': False
        },
        {
            'id': 2,
            'type': 'trend',
            'priority': 'warning',
            'title': 'Deteksi Pembalikan Trend',
            'message': 'Trend berubah dari naik ke turun dalam 3 periode terakhir',
            'timestamp': '2024-01-20T12:15:00',
            'acknowledged': True
        },
        {
            'id': 3,
            'type': 'volatility',
            'priority': 'info',
            'title': 'Volatilitas Meningkat',
            'message': 'Volatilitas 7 hari meningkat 25% dari rata-rata',
            'timestamp': '2024-01-20T10:00:00',
            'acknowledged': False
        }
    ]
    
    return jsonify({
        'success': True,
        'alerts': alerts
    })

@app.route('/api/alerts/history')
def api_alert_history():
    """Get historical alert records (not time series chart)"""
    try:
        days = int(request.args.get('days', 7))
        result = forecast_service.get_historical_alerts(days)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error getting alert history: {str(e)}',
            'alerts': []
        })
        
@app.route('/api/alerts/statistical')
def api_statistical_alerts():
    """Get real-time statistical alerts"""
    try:
        result = forecast_service.get_statistical_alerts()
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'alerts': []})
    
def filter_by_timeframe(df, timeframe):
    """Filter dataframe by timeframe"""
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    
    if timeframe == '1M':
        cutoff = datetime.now() - timedelta(days=30)
    elif timeframe == '3M':
        cutoff = datetime.now() - timedelta(days=90)
    elif timeframe == '6M':
        cutoff = datetime.now() - timedelta(days=180)
    elif timeframe == '1Y':
        cutoff = datetime.now() - timedelta(days=365)
    else:  # ALL
        return df
    
    return df[df['Tanggal'] >= cutoff]

@app.route('/commodity-insights')
def commodity_insights():
    """Commodity Insights page"""
    return render_template('commodity_insights.html', page_title="Commodity Insights")

# API endpoints untuk commodity insights
@app.route('/api/commodity/current-week')
def api_commodity_current_week():
    """FIXED - Enhanced current week commodity insights"""
    try:
        print("🔍 API: Loading current week insights...")
        result = commodity_service.get_current_week_insights()
        
        print(f"📊 Current week result structure: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        print(f"📊 Success status: {result.get('success')}")
        
        if result.get('success'):
            print(f"   📅 Period keys: {list(result.get('period', {}).keys())}")
            print(f"   📈 IPH analysis keys: {list(result.get('iph_analysis', {}).keys())}")
            print(f"   🏷️ Category analysis count: {len(result.get('category_analysis', {}))}")
        else:
            print(f"   ❌ Error: {result.get('message', 'Unknown error')}")
        
        # Ensure all required fields exist
        if result.get('success') and not result.get('iph_analysis'):
            print("⚠️ Missing iph_analysis, creating fallback...")
            iph_value = result.get('iph_value', 0)
            result['iph_analysis'] = {
                'value': float(iph_value),
                'level': 'Unknown',
                'color': 'secondary',
                'direction': 'Unknown'
            }
        
        return jsonify(clean_for_json(result))
        
    except Exception as e:
        print(f"❌ API Error - current week insights: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify(clean_for_json({
            'success': False, 
            'error': str(e),
            'message': 'Failed to load current week insights. Please check if commodity data is available.',
            'error_details': str(e)
        }))

@app.route('/api/commodity/monthly-analysis')
def api_commodity_monthly():
    """FIXED - Enhanced monthly commodity analysis"""
    try:
        month = request.args.get('month', '').strip()
        print(f"🔍 API: Loading monthly analysis for month: '{month}'")
        
        result = commodity_service.get_monthly_analysis(month if month else None)
        
        print(f"📊 Monthly analysis result structure: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        print(f"📊 Success: {result.get('success')}")
        
        if result.get('success'):
            print(f"   📅 Month: {result.get('month')}")
            print(f"   📊 Analysis period keys: {list(result.get('analysis_period', {}).keys())}")
            print(f"   📈 IPH stats keys: {list(result.get('iph_statistics', {}).keys())}")
        else:
            print(f"   ❌ Error: {result.get('message')}")
        
        return jsonify(clean_for_json(result))
        
    except Exception as e:
        print(f"❌ API Error - monthly analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify(clean_for_json({
            'success': False, 
            'error': str(e),
            'message': 'Failed to load monthly analysis. Please check if commodity data is available.',
            'error_type': 'processing_error'
        }))

@app.route('/api/commodity/trends')
def api_commodity_trends():
    """FIXED - Enhanced commodity trends"""
    try:
        commodity = request.args.get('commodity', '').strip()
        periods = int(request.args.get('periods', 4))
        
        # Validate periods
        if not (2 <= periods <= 24):
            periods = 4
        
        print(f"🔍 API: Loading commodity trends - periods: {periods}, commodity: '{commodity}'")
        
        result = commodity_service.get_commodity_trends(
            commodity if commodity else None, 
            periods
        )
        
        print(f"📊 Trends result structure: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        print(f"📊 Success: {result.get('success')}")
        
        if result.get('success'):
            trends_count = len(result.get('commodity_trends', {}))
            print(f"   📈 Found {trends_count} commodity trends")
            
            # Debug first few trends
            if result.get('commodity_trends'):
                first_trend = list(result['commodity_trends'].items())[0] if result['commodity_trends'] else None
                if first_trend:
                    trend_name, trend_data = first_trend
                    print(f"   🔍 First trend '{trend_name}' keys: {list(trend_data.keys())}")
        
        return jsonify(clean_for_json(result))
        
    except Exception as e:
        print(f"❌ API Error - commodity trends: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify(clean_for_json({
            'success': False, 
            'error': str(e),
            'message': 'Failed to load commodity trends'
        }))

@app.route('/api/commodity/seasonal')
def api_commodity_seasonal():
    """FIXED - Enhanced seasonal commodity patterns"""
    try:
        print("🔍 API: Loading seasonal patterns...")
        
        result = commodity_service.get_seasonal_patterns()
        
        print(f"📊 Seasonal result structure: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        print(f"📊 Success: {result.get('success')}")
        
        if result.get('success'):
            patterns_count = len(result.get('seasonal_patterns', {}))
            print(f"   🗓️ Found {patterns_count} monthly patterns")
            
            # Debug structure
            if result.get('seasonal_patterns'):
                first_pattern = list(result['seasonal_patterns'].items())[0] if result['seasonal_patterns'] else None
                if first_pattern:
                    month_name, month_data = first_pattern
                    print(f"   🔍 First pattern '{month_name}' keys: {list(month_data.keys())}")
        
        return jsonify(clean_for_json(result))
        
    except Exception as e:
        print(f"❌ API Error - seasonal patterns: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify(clean_for_json({
            'success': False, 
            'error': str(e),
            'message': 'Failed to load seasonal patterns'
        }))

@app.route('/api/commodity/alerts')
def api_commodity_alerts():
    """Enhanced commodity volatility alerts - FIXED VERSION"""
    try:
        threshold = float(request.args.get('threshold', 0.05))
        
        # Validate threshold
        if not (0.01 <= threshold <= 0.5):
            threshold = 0.05
        
        print(f"🔍 API: Loading volatility alerts with threshold: {threshold}")
        
        result = commodity_service.get_alert_commodities(threshold)
        
        print(f"📊 Alerts result: success={result.get('success')}")
        if result.get('success'):
            alerts_count = len(result.get('alerts', []))
            print(f"   ⚠️ Found {alerts_count} alerts")
        
        return jsonify(clean_for_json(result))
    except Exception as e:
        print(f"❌ API Error - commodity alerts: {str(e)}")
        return jsonify(clean_for_json({
            'success': False, 
            'error': str(e),
            'message': 'Failed to load commodity alerts'
        }))

@app.route('/api/commodity/upload', methods=['POST'])
def upload_commodity_data():
    """Enhanced commodity data upload with comprehensive validation - FIXED VERSION"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'})
        
        if file and file.filename.lower().endswith(('.csv', '.xlsx')):
            # Save temporarily for processing
            filename = secure_filename(file.filename)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_commodity_{filename}")
            file.save(temp_path)
            
            try:
                print(f"📂 Processing commodity file: {filename}")
                
                # Enhanced file reading with multiple encoding support
                if filename.lower().endswith('.csv'):
                    df = None
                    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                        try:
                            df = pd.read_csv(temp_path, encoding=encoding)
                            print(f"✅ CSV loaded successfully with {encoding} encoding")
                            break
                        except UnicodeDecodeError:
                            print(f"⚠️ Failed to load with {encoding} encoding, trying next...")
                            continue
                    
                    if df is None:
                        return jsonify({
                            'success': False, 
                            'message': 'Unable to read CSV file with any encoding. Please save as UTF-8 CSV.'
                        })
                else:
                    df = pd.read_excel(temp_path)
                    print("✅ Excel file loaded successfully")
                
                print(f"📊 Loaded {len(df)} rows, {len(df.columns)} columns")
                print(f"📋 Original columns: {list(df.columns)}")
                
                # Enhanced column validation with fuzzy matching
                required_column_patterns = {
                    'bulan': [r'.*[Bb]ulan.*', r'.*[Mm]onth.*'],
                    'minggu': [r'.*[Mm]inggu.*', r'.*[Ww]eek.*'],
                    'iph': [r'.*[Ii]ndikator.*[Pp]erubahan.*[Hh]arga.*', r'.*IPH.*', r'.*iph.*'],
                    'komoditas_andil': [r'.*[Kk]omoditas.*[Aa]ndil.*', r'.*[Cc]ommodity.*[Ii]mpact.*'],
                    'komoditas_fluktuasi': [r'.*[Kk]omoditas.*[Ff]luktuasi.*', r'.*[Vv]olatile.*[Cc]ommodity.*'],
                    'nilai_fluktuasi': [r'.*[Ff]luktuasi.*[Hh]arga.*', r'.*[Vv]olatility.*[Vv]alue.*']
                }
                
                column_mapping = {}
                missing_requirements = []
                
                for req_type, patterns in required_column_patterns.items():
                    found = False
                    for pattern in patterns:
                        for col in df.columns:
                            if re.match(pattern, col, re.IGNORECASE):
                                column_mapping[col] = req_type
                                found = True
                                break
                        if found:
                            break
                    
                    if not found and req_type in ['bulan', 'minggu', 'iph']:  # Only critical columns
                        missing_requirements.append(req_type)
                
                if missing_requirements:
                    return jsonify({
                        'success': False,
                        'message': f'Missing critical columns: {", ".join(missing_requirements)}',
                        'available_columns': list(df.columns),
                        'required_patterns': {k: v[0] for k, v in required_column_patterns.items() if k in missing_requirements}
                    })
                
                print(f"✅ Column mapping successful: {column_mapping}")
                
                # Apply basic data cleaning
                df = df.dropna(how='all')  # Remove completely empty rows
                
                # Backup existing data if requested
                commodity_path = commodity_service.commodity_data_path
                if request.form.get('backup_existing') == 'true':
                    if os.path.exists(commodity_path):
                        backup_path = commodity_path.replace('.csv', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
                        try:
                            import shutil
                            shutil.copy2(commodity_path, backup_path)
                            print(f"📦 Existing data backed up to: {backup_path}")
                        except Exception as backup_error:
                            print(f"⚠️ Backup failed: {backup_error}")
                
                # Save new data
                os.makedirs(os.path.dirname(commodity_path), exist_ok=True)
                df.to_csv(commodity_path, index=False)
                
                # Clear service cache to force reload
                commodity_service.commodity_cache = None
                commodity_service.last_cache_time = None
                
                print(f"✅ Commodity data saved to: {commodity_path}")
                
                # Validate the saved data by trying to load it
                try:
                    test_df = commodity_service.load_commodity_data()
                    if test_df.empty:
                        print("⚠️ Warning: Saved data appears to be empty after processing")
                except Exception as validation_error:
                    print(f"⚠️ Data validation warning: {validation_error}")
                
                return jsonify(clean_for_json({
                    'success': True,
                    'message': 'Commodity data uploaded and processed successfully',
                    'records': len(df),
                    'columns_mapped': len(column_mapping),
                    'original_columns': list(df.columns),
                    'processing_info': {
                        'empty_rows_removed': 'yes',
                        'encoding_used': 'auto-detected',
                        'backup_created': request.form.get('backup_existing') == 'true'
                    }
                }))
                
            except Exception as processing_error:
                print(f"❌ Processing error: {str(processing_error)}")
                import traceback
                traceback.print_exc()
                
                return jsonify({
                    'success': False, 
                    'message': f'File processing failed: {str(processing_error)}',
                    'error_type': 'processing_error'
                })
                
            finally:
                # Clean up temp file
                try:
                    os.remove(temp_path)
                    print(f"🗑️ Cleaned up temp file: {temp_path}")
                except:
                    pass
        
        return jsonify({
            'success': False, 
            'message': 'Invalid file format. Please upload CSV or Excel file.',
            'allowed_formats': ['.csv', '.xlsx']
        })
        
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify(clean_for_json({
            'success': False, 
            'message': f'Upload failed: {str(e)}',
            'error_type': type(e).__name__
        }))

@app.route('/api/commodity/data-status')
def api_commodity_data_status():
    """Check commodity data availability"""
    try:
        df = commodity_service.load_commodity_data()
        
        return jsonify(clean_for_json({
            'success': True,
            'has_data': not df.empty,
            'record_count': len(df) if not df.empty else 0,
            'date_range': {
                'start': df['Tanggal'].min().strftime('%Y-%m-%d') if not df.empty and 'Tanggal' in df.columns else None,
                'end': df['Tanggal'].max().strftime('%Y-%m-%d') if not df.empty and 'Tanggal' in df.columns else None
            } if not df.empty else None,
            'columns': list(df.columns) if not df.empty else [],
            'last_updated': datetime.now().isoformat()
        }))
    except Exception as e:
        print(f"❌ Commodity data status error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'has_data': False,
            'record_count': 0
        })


    """Enhanced current week commodity insights"""
    try:
        result = commodity_service.get_current_week_insights()
        return jsonify(clean_for_json(result))
    except Exception as e:
        print(f"❌ API Error - current week: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False, 
            'error': str(e),
            'message': 'Failed to load current week insights',
            'error_details': str(e)
        })


app.add_url_rule('/upload', 'data_control', data_control)

# Di bagian akhir app.py, ganti bagian if __name__ == '__main__':
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV', 'production') == 'development'
    
    # Untuk development
    if debug_mode:
        app.run(host='0.0.0.0', port=port, debug=True)
    else:
        # Untuk production, gunakan gunicorn
        print(f"🚀 Starting production server on port {port}")
        print("📊 Dashboard will be available at: http://0.0.0.0:" + str(port))
        app.run(host='0.0.0.0', port=port, debug=False)