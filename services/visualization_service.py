import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.utils
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

class VisualizationService:
    """Service for advanced data visualization and analysis"""
    
    def __init__(self, data_handler):
        self.data_handler = data_handler
    
    def filter_by_timeframe(self, df, timeframe):
        """Filter dataframe by timeframe"""
        if df.empty:
            return df
            
        df = df.copy()
        df['Tanggal'] = pd.to_datetime(df['Tanggal'])
        
        if timeframe == '6M':
            cutoff = datetime.now() - timedelta(days=180)
        elif timeframe == '1Y':
            cutoff = datetime.now() - timedelta(days=365)
        else:  # ALL
            return df
        
        return df[df['Tanggal'] >= cutoff].reset_index(drop=True)
    
    def perform_decomposition(self, timeframe='6M'):
        """Perform time series decomposition with improved error handling"""
        try:
            # Load data
            df = self.data_handler.load_historical_data()
            if df.empty:
                return {'success': False, 'message': 'Tidak ada data tersedia'}
            
            # Filter by timeframe
            df = self.filter_by_timeframe(df, timeframe)
            
            # Ensure we have enough data
            if len(df) < 12:
                return {'success': False, 'message': 'Data tidak cukup untuk dekomposisi (minimal 12 data point)'}
            
            # Clean data
            df = df.dropna(subset=['Tanggal', 'Indikator_Harga'])
            df['Tanggal'] = pd.to_datetime(df['Tanggal'])
            df = df.sort_values('Tanggal').reset_index(drop=True)
            
            # Simple decomposition using moving averages
            window_size = min(7, len(df) // 3)
            
            # Calculate trend using moving average
            df['Trend'] = df['Indikator_Harga'].rolling(window=window_size, center=True).mean()
            df['Trend'].fillna(method='bfill', inplace=True)
            df['Trend'].fillna(method='ffill', inplace=True)
            
            # Calculate detrended series
            df['Detrended'] = df['Indikator_Harga'] - df['Trend']
            
            # Calculate seasonal component
            if len(df) >= 12:
                df['Week'] = df['Tanggal'].dt.isocalendar().week
                seasonal_avg = df.groupby('Week')['Detrended'].transform('mean')
                df['Seasonal'] = seasonal_avg
            else:
                x = np.arange(len(df))
                df['Seasonal'] = np.sin(2 * np.pi * x / window_size) * df['Detrended'].std()
            
            # Calculate residual
            df['Residual'] = df['Indikator_Harga'] - df['Trend'] - df['Seasonal']
            
            # Create figure with subplots
            from plotly.subplots import make_subplots
            
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=('Data Asli', 'Trend', 'Seasonal', 'Residual'),
                vertical_spacing=0.08
            )

            # Add traces with proper data cleaning
            def clean_series(series):
                """Clean series for plotly"""
                return series.fillna(0).replace([np.inf, -np.inf], 0)

            # Add traces
            fig.add_trace(
                go.Scatter(
                    x=df['Tanggal'],
                    y=clean_series(df['Indikator_Harga']),
                    mode='lines',
                    name='Data Asli',
                    line=dict(color='#667eea', width=2)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['Tanggal'],
                    y=clean_series(df['Trend']),
                    mode='lines',
                    name='Trend',
                    line=dict(color='#f093fb', width=2)
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['Tanggal'],
                    y=clean_series(df['Seasonal']),
                    mode='lines',
                    name='Seasonal',
                    line=dict(color='#4facfe', width=2)
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['Tanggal'],
                    y=clean_series(df['Residual']),
                    mode='lines',
                    name='Residual',
                    line=dict(color='#fa709a', width=2)
                ),
                row=4, col=1
            )
            
            # Update layout
            fig.update_layout(
                title='Dekomposisi Time Series IPH',
                height=600,
                showlegend=False,
                template='plotly_white'
            )

            # Return as proper dictionary, not JSON string
            return {
                'success': True,
                'chart': {
                    'data': [
                        {
                            'x': [d.isoformat() if hasattr(d, 'isoformat') else str(d) for d in df['Tanggal']],
                            'y': [float(v) if not pd.isna(v) else 0 for v in clean_series(df['Indikator_Harga'])],
                            'mode': 'lines',
                            'name': 'Data Asli',
                            'line': {'color': '#667eea', 'width': 2},
                            'yaxis': 'y1'
                        },
                        {
                            'x': [d.isoformat() if hasattr(d, 'isoformat') else str(d) for d in df['Tanggal']],
                            'y': [float(v) if not pd.isna(v) else 0 for v in clean_series(df['Trend'])],
                            'mode': 'lines',
                            'name': 'Trend',
                            'line': {'color': '#f093fb', 'width': 2},
                            'yaxis': 'y2'
                        },
                        {
                            'x': [d.isoformat() if hasattr(d, 'isoformat') else str(d) for d in df['Tanggal']],
                            'y': [float(v) if not pd.isna(v) else 0 for v in clean_series(df['Seasonal'])],
                            'mode': 'lines',
                            'name': 'Seasonal',
                            'line': {'color': '#4facfe', 'width': 2},
                            'yaxis': 'y3'
                        },
                        {
                            'x': [d.isoformat() if hasattr(d, 'isoformat') else str(d) for d in df['Tanggal']],
                            'y': [float(v) if not pd.isna(v) else 0 for v in clean_series(df['Residual'])],
                            'mode': 'lines',
                            'name': 'Residual',
                            'line': {'color': '#fa709a', 'width': 2},
                            'yaxis': 'y4'
                        }
                    ],
                    'layout': {
                        'title': 'Dekomposisi Time Series IPH',
                        'height': 600,
                        'showlegend': False,
                        'template': 'plotly_white',
                        'grid': {'rows': 4, 'columns': 1, 'subplots': [['xy'], ['xy2'], ['xy3'], ['xy4']]},
                        'yaxis': {'domain': [0.775, 1.0], 'title': 'Original'},
                        'yaxis2': {'domain': [0.525, 0.725], 'title': 'Trend'},
                        'yaxis3': {'domain': [0.275, 0.475], 'title': 'Seasonal'},
                        'yaxis4': {'domain': [0.0, 0.225], 'title': 'Residual'},
                        'xaxis4': {'title': 'Date'}
                    }
                }
            }
            
        except Exception as e:
            print(f"Error in decomposition: {str(e)}")
            return {'success': False, 'message': f'Error: {str(e)}'}
        
    def calculate_moving_averages(self, timeframe='6M'):
        """Calculate various moving averages with proper JSON serialization"""
        try:

            print(f"📊 Moving averages - timeframe: {timeframe}")
        
            df = self.data_handler.load_historical_data()
            print(f"   📋 Data loaded: {len(df) if not df.empty else 0} records")
            
            if df.empty:
                return {'success': False, 'message': 'Tidak ada data tersedia'}

            df = self.data_handler.load_historical_data()
            if df.empty:
                return {'success': False, 'message': 'Tidak ada data tersedia'}

            df = self.filter_by_timeframe(df, timeframe)
            df = df.sort_values('Tanggal').reset_index(drop=True)
            
            if len(df) < 7:
                return {'success': False, 'message': 'Data tidak cukup untuk moving averages (minimal 7 data)'}

            # Calculate Simple Moving Averages
            df['SMA_7'] = df['Indikator_Harga'].rolling(window=7, min_periods=1).mean()
            df['SMA_14'] = df['Indikator_Harga'].rolling(window=14, min_periods=1).mean()
            df['SMA_30'] = df['Indikator_Harga'].rolling(window=30, min_periods=1).mean()
            
            # Calculate Exponential Moving Averages
            df['EMA_7'] = df['Indikator_Harga'].ewm(span=7, adjust=False).mean()
            df['EMA_14'] = df['Indikator_Harga'].ewm(span=14, adjust=False).mean()
            
            # Calculate Weighted Moving Average
            def wma(data, period):
                weights = np.arange(1, period + 1)
                return data.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum() if len(x) == period else x.iloc[-1], raw=False)
            
            df['WMA_7'] = wma(df['Indikator_Harga'], 7)
            df['WMA_14'] = wma(df['Indikator_Harga'], 14)
            
            # Clean data function
            def clean_series(series):
                return series.fillna(0).replace([np.inf, -np.inf], 0)
            
            # Prepare dates as strings
            dates = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in df['Tanggal']]
            
            # Create chart data with proper structure
            chart_data = {
                'data': [
                    {
                        'x': dates,
                        'y': [float(v) if not pd.isna(v) else 0 for v in clean_series(df['Indikator_Harga'])],
                        'mode': 'lines',
                        'name': 'IPH Aktual',
                        'line': {'color': '#1f77b4', 'width': 3},
                        'visible': True
                    },
                    {
                        'x': dates,
                        'y': [float(v) if not pd.isna(v) else 0 for v in clean_series(df['SMA_7'])],
                        'mode': 'lines',
                        'name': 'SMA 7',
                        'line': {'color': '#ff7f0e', 'width': 2},
                        'visible': True
                    },
                    {
                        'x': dates,
                        'y': [float(v) if not pd.isna(v) else 0 for v in clean_series(df['SMA_14'])],
                        'mode': 'lines',
                        'name': 'SMA 14',
                        'line': {'color': '#2ca02c', 'width': 2},
                        'visible': True
                    },
                    {
                        'x': dates,
                        'y': [float(v) if not pd.isna(v) else 0 for v in clean_series(df['SMA_30'])],
                        'mode': 'lines',
                        'name': 'SMA 30',
                        'line': {'color': '#d62728', 'width': 2},
                        'visible': True
                    },
                    {
                        'x': dates,
                        'y': [float(v) if not pd.isna(v) else 0 for v in clean_series(df['EMA_7'])],
                        'mode': 'lines',
                        'name': 'EMA 7',
                        'line': {'color': '#9467bd', 'width': 2, 'dash': 'dash'},
                        'visible': False
                    },
                    {
                        'x': dates,
                        'y': [float(v) if not pd.isna(v) else 0 for v in clean_series(df['EMA_14'])],
                        'mode': 'lines',
                        'name': 'EMA 14',
                        'line': {'color': '#8c564b', 'width': 2, 'dash': 'dash'},
                        'visible': False
                    },
                    {
                        'x': dates,
                        'y': [float(v) if not pd.isna(v) else 0 for v in clean_series(df['WMA_7'])],
                        'mode': 'lines',
                        'name': 'WMA 7',
                        'line': {'color': '#e377c2', 'width': 2, 'dash': 'dot'},
                        'visible': False
                    },
                    {
                        'x': dates,
                        'y': [float(v) if not pd.isna(v) else 0 for v in clean_series(df['WMA_14'])],
                        'mode': 'lines',
                        'name': 'WMA 14',
                        'line': {'color': '#7f7f7f', 'width': 2, 'dash': 'dot'},
                        'visible': False
                    }
                ],
                'layout': {
                    'title': 'Analisis Moving Averages IPH',
                    'xaxis': {'title': 'Tanggal'},
                    'yaxis': {'title': 'IPH (%)'},
                    'hovermode': 'x unified',
                    'height': 500,
                    'template': 'plotly_white',
                    'showlegend': True
                }
            }
            
            return {
                'success': True,
                'chart': chart_data
            }
            
        except Exception as e:
            print(f"Error in moving averages: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
            'success': True,
            'chart': chart_data
        }
   

    def analyze_volatility(self, timeframe='6M'):
        """Analyze price volatility"""
        try:
            df = self.data_handler.load_historical_data()
            if df.empty:
                return {'success': False, 'message': 'Tidak ada data tersedia'}
            
            df = self.filter_by_timeframe(df, timeframe)
            df = df.sort_values('Tanggal').reset_index(drop=True)
            
            # Calculate rolling volatility (standard deviation)
            df['Volatility_7'] = df['Indikator_Harga'].rolling(window=7, min_periods=1).std()
            df['Volatility_14'] = df['Indikator_Harga'].rolling(window=14, min_periods=1).std()
            df['Volatility_30'] = df['Indikator_Harga'].rolling(window=30, min_periods=1).std()
            
            # Calculate average volatility line
            avg_volatility = df['Volatility_7'].mean()
            
            # Create figure
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df['Tanggal'],
                y=df['Volatility_7'],
                mode='lines',
                name='Volatilitas 7 Hari',
                fill='tozeroy',
                fillcolor='rgba(239, 68, 68, 0.3)',
                fillpattern=dict(
                    shape=".",
                    bgcolor="rgba(239, 68, 68, 0.1)",
                    fgcolor="rgba(239, 68, 68, 0.5)",
                    size=8,
                    solidity=0.2
                ),
                line=dict(
                    color='rgba(239, 68, 68, 1)',
                    width=2,
                    shape='spline',
                    smoothing=0.3
                )
            ))
            
            fig.add_trace(go.Scatter(
                x=df['Tanggal'],
                y=df['Volatility_14'],
                mode='lines',
                name='Volatilitas 14 Hari',
                fill='tozeroy',
                fillcolor='rgba(239, 68, 68, 0.3)',
                fillpattern=dict(
                    shape=".",
                    bgcolor="rgba(239, 68, 68, 0.1)",
                    fgcolor="rgba(239, 68, 68, 0.5)",
                    size=8,
                    solidity=0.2
                ),
                line=dict(
                    color='rgba(239, 68, 68, 1)',
                    width=2,
                    shape='spline',
                    smoothing=0.3
                )
            ))
            
            fig.add_trace(go.Scatter(
                x=df['Tanggal'],
                y=df['Volatility_30'],
                mode='lines',
                name='Volatilitas 30 Hari',
                fill='tozeroy',
                fillcolor='rgba(239, 68, 68, 0.3)',
                fillpattern=dict(
                    shape=".",
                    bgcolor="rgba(239, 68, 68, 0.1)",
                    fgcolor="rgba(239, 68, 68, 0.5)",
                    size=8,
                    solidity=0.2
                ),
                line=dict(
                    color='rgba(239, 68, 68, 1)',
                    width=2,
                    shape='spline',
                    smoothing=0.3
                )
            ))
            
            fig.add_trace(go.Scatter(
                x=[df['Tanggal'].iloc[0], df['Tanggal'].iloc[-1]],
                y=[avg_volatility, avg_volatility],
                mode='lines',
                name='Rata-rata Volatilitas',
                fill='tozeroy',
                fillcolor='rgba(239, 68, 68, 0.3)',
                fillpattern=dict(
                    shape=".",
                    bgcolor="rgba(239, 68, 68, 0.1)",
                    fgcolor="rgba(239, 68, 68, 0.5)",
                    size=8,
                    solidity=0.2
                ),
                line=dict(
                    color='rgba(239, 68, 68, 1)',
                    width=2,
                    shape='spline',
                    smoothing=0.3
                )
            ))
            
            fig.update_layout(
                title='Analisis Volatilitas IPH (Rolling Standard Deviation)',
                xaxis_title='Tanggal',
                yaxis_title='Volatilitas (%)',
                hovermode='x unified',
                height=400,
                template='plotly_white'
            )
            
            # Calculate statistics
            current_vol = df['Volatility_7'].iloc[-1] if len(df) > 0 else 0
            avg_30 = df['Volatility_30'].mean() if len(df) > 0 else 0
            max_vol = df['Volatility_7'].max() if len(df) > 0 else 0
            min_vol = df['Volatility_7'].min() if len(df) > 0 else 0
            
            stats = {
                'current': float(current_vol) if not pd.isna(current_vol) else 0,
                'avg30': float(avg_30) if not pd.isna(avg_30) else 0,
                'max': float(max_vol) if not pd.isna(max_vol) else 0,
                'min': float(min_vol) if not pd.isna(min_vol) else 0
            }
            
            # Convert to JSON serializable format
            fig_json = plotly.utils.PlotlyJSONEncoder().encode(fig)
            
            return {
                'success': True,
                'chart': fig_json,
                'stats': stats
            }
            
        except Exception as e:
            print(f"Error in volatility analysis: {str(e)}")
            return {'success': False, 'message': str(e)}
    
    def analyze_distribution(self, timeframe='6M'):
        """Analyze price change distribution"""
        try:
            df = self.data_handler.load_historical_data()
            if df.empty:
                return {'success': False, 'message': 'Tidak ada data tersedia'}
            
            df = self.filter_by_timeframe(df, timeframe)
            
            # Categorize price changes
            bins = [-float('inf'), -3, -1, 0, 1, 3, float('inf')]
            labels = ['Turun Tajam\n(<-3%)', 'Turun Moderat\n(-3% to -1%)', 
                     'Turun Ringan\n(-1% to 0%)', 'Naik Ringan\n(0% to 1%)', 
                     'Naik Moderat\n(1% to 3%)', 'Naik Tajam\n(>3%)']
            
            df['Category'] = pd.cut(df['Indikator_Harga'], bins=bins, labels=labels)
            category_counts = df['Category'].value_counts()
            
            # Pie chart
            pie_fig = go.Figure(data=[go.Pie(
                labels=category_counts.index.tolist(),
                values=category_counts.values.tolist(),
                hole=0.3,
                marker=dict(colors=['#d62728', '#ff7f0e', '#ffbb78', 
                                  '#98df8a', '#2ca02c', '#1f77b4']),
                textposition='inside',
                textinfo='percent+label'
            )])
            
            pie_fig.update_layout(
                title='Distribusi Perubahan Harga',
                height=400,
                template='plotly_white'
            )
            
            # Histogram
            hist_fig = go.Figure()
            
            hist_fig.add_trace(go.Histogram(
                x=df['Indikator_Harga'],
                nbinsx=30,
                marker=dict(color='rgba(102, 126, 234, 0.7)', 
                           line=dict(color='rgba(102, 126, 234, 1)', width=1)),
                name='Frekuensi'
            ))
            
            # Add normal distribution overlay
            mean_val = df['Indikator_Harga'].mean()
            std_val = df['Indikator_Harga'].std()
            x_range = np.linspace(df['Indikator_Harga'].min(), df['Indikator_Harga'].max(), 100)
            normal_dist = ((1 / (std_val * np.sqrt(2 * np.pi))) * 
                          np.exp(-0.5 * ((x_range - mean_val) / std_val) ** 2))
            
            # Scale normal distribution to match histogram
            hist_values, bin_edges = np.histogram(df['Indikator_Harga'], bins=30)
            scale_factor = len(df) * (bin_edges[1] - bin_edges[0])
            
            hist_fig.add_trace(go.Scatter(
                x=x_range,
                y=normal_dist * scale_factor,
                mode='lines',
                name='Distribusi Normal',
                line=dict(color='red', width=2),
                yaxis='y2'
            ))
            
            hist_fig.update_layout(
                title='Histogram Perubahan IPH',
                xaxis_title='IPH (%)',
                yaxis_title='Frekuensi',
                yaxis2=dict(overlaying='y', side='right', title='Densitas'),
                height=400,
                template='plotly_white',
                bargap=0.1
            )
            
            # Convert to JSON serializable format
            pie_json = plotly.utils.PlotlyJSONEncoder().encode(pie_fig)
            hist_json = plotly.utils.PlotlyJSONEncoder().encode(hist_fig)
            
            return {
                'success': True,
                'pieChart': pie_json,
                'histogram': hist_json
            }
            
        except Exception as e:
            print(f"Error in distribution analysis: {str(e)}")
            return {'success': False, 'message': str(e)}
    
    def analyze_aggregation(self, agg_type='monthly', timeframe='6M'):
        """Analyze data with different aggregation periods"""
        try:
            df = self.data_handler.load_historical_data()
            if df.empty:
                return {'success': False, 'message': 'Tidak ada data tersedia'}
            
            df = self.filter_by_timeframe(df, timeframe)
            df['Tanggal'] = pd.to_datetime(df['Tanggal'])
            df = df.sort_values('Tanggal')
            
            # Perform aggregation
            if agg_type == 'monthly':
                df['Period'] = df['Tanggal'].dt.to_period('M')
                title = 'Agregasi Bulanan IPH'
            elif agg_type == 'quarterly':
                df['Period'] = df['Tanggal'].dt.to_period('Q')
                title = 'Agregasi Kuartalan IPH'
            else:  # yearly
                df['Period'] = df['Tanggal'].dt.to_period('Y')
                title = 'Agregasi Tahunan IPH'
            
            # Group and aggregate
            agg_df = df.groupby('Period').agg({
                'Indikator_Harga': ['mean', 'min', 'max', 'std', 'count']
            }).reset_index()
            
            # Flatten column names
            agg_df.columns = ['Period', 'Mean', 'Min', 'Max', 'Std', 'Count']
            agg_df['Period_str'] = agg_df['Period'].astype(str)
            
            # Create figure with secondary y-axis
            from plotly.subplots import make_subplots
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add bar chart for mean values
            fig.add_trace(
                go.Bar(
                    name='Rata-rata',
                    x=agg_df['Period_str'],
                    y=agg_df['Mean'],
                    marker_color='rgba(102, 126, 234, 0.7)',
                    text=[f'{val:.2f}%' for val in agg_df['Mean']],
                    textposition='auto'
                ),
                secondary_y=False
            )
            
            # Add line charts for min/max
            fig.add_trace(
                go.Scatter(
                    x=agg_df['Period_str'],
                    y=agg_df['Max'],
                    mode='lines+markers',
                    name='Maksimum',
                    line=dict(color='red', width=2),
                    marker=dict(size=8)
                ),
                secondary_y=True
            )
            
            fig.add_trace(
                go.Scatter(
                    x=agg_df['Period_str'],
                    y=agg_df['Min'],
                    mode='lines+markers',
                    name='Minimum',
                    line=dict(color='green', width=2),
                    marker=dict(size=8)
                ),
                secondary_y=True
            )
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title='Periode',
                hovermode='x unified',
                height=450,
                template='plotly_white',
                bargap=0.2
            )
            
            # Set y-axes titles
            fig.update_yaxes(title_text="IPH Rata-rata (%)", secondary_y=False)
            fig.update_yaxes(title_text="Min/Max (%)", secondary_y=True)
            
            # Convert to JSON serializable format
            fig_json = plotly.utils.PlotlyJSONEncoder().encode(fig)
            
            return {
                'success': True,
                'chart': fig_json
            }
            
        except Exception as e:
            print(f"Error in aggregation analysis: {str(e)}")
            return {'success': False, 'message': str(e)}