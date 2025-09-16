// IPH Forecasting Dashboard JavaScript
let modelMetricsCache = null;
let modelMetricsCacheTime = 0;

class ForecastingDashboard {
    constructor() {
        this.charts = {};
        this.refreshInterval = null;
        this.init();
    }

    init() {
        console.log('🚀 Initializing Forecasting Dashboard...');
        
        // Initialize event listeners
        this.initEventListeners();
        
        // Load initial data
        this.loadDashboardData();
        
        // Setup auto-refresh
        this.setupAutoRefresh();
        
        setInterval(() => {
            this.updateRealTimeStatus();
        }, 1000); 

        console.log('✅ Dashboard initialized successfully');
    }

    initEventListeners() {
        // File upload handling
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const browseBtn = document.getElementById('browseBtn');
        
        if (uploadArea && fileInput) {
            // Drag and drop
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    this.handleFileUpload(files[0]);
                }
            });
            
            // Browse button click
            if (browseBtn) {
                browseBtn.addEventListener('click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    fileInput.click();
                });
            }
            
            // File input change
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    this.handleFileUpload(e.target.files[0]);
                }
            });
        }
    
        // Other event listeners...
        
        const generateForecastBtn = document.getElementById('generateForecastBtn');
        if (generateForecastBtn) {
            generateForecastBtn.addEventListener('click', () => {
                this.showForecastModal();
            });
        }
    
        const retrainModelsBtn = document.getElementById('retrainModelsBtn');
        if (retrainModelsBtn) {
            retrainModelsBtn.addEventListener('click', () => {
                this.retrainModels();
            });
        }
    
        const refreshBtn = document.getElementById('refreshBtn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                this.refreshDashboard();
            });
        }
    }

    async handleFileUpload(file) {
        console.log(`📁 Uploading file: ${file.name}`);
        
        // Validate file type
        const allowedTypes = ['text/csv', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'];
        if (!allowedTypes.includes(file.type)) {
            this.showAlert('Please upload a CSV or Excel file.', 'warning');
            return;
        }

        // Validate file size (16MB max)
        if (file.size > 16 * 1024 * 1024) {
            this.showAlert('File size must be less than 16MB.', 'warning');
            return;
        }

        // Show loading
        this.showLoading('Processing data and training models...');

        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('forecast_weeks', document.getElementById('forecastWeeks')?.value || '8');

            const response = await fetch('/api/upload-data', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            
            if (result.success) {
                this.showAlert('Data processed successfully! Models trained and forecast generated.', 'success');
                this.displayUploadResults(result);
                
                // Refresh dashboard after successful upload
                setTimeout(() => {
                    this.refreshDashboard();
                }, 2000);
            } else {
                this.showAlert(`Error: ${result.error || result.message}`, 'danger');
            }

        } catch (error) {
            console.error('Upload error:', error);
            this.showAlert(`Upload failed: ${error.message}`, 'danger');
        } finally {
            this.hideLoading();
        }
    }

    async generateForecast(modelName, weeks) {
        console.log("=" .repeat(80));
        console.log(`🔮 GENERATE FORECAST FRONTEND DEBUG:`);
        console.log(`   📊 Weeks: ${weeks}`);
        console.log(`   🤖 Model: '${modelName}'`);
        console.log(`   📋 Model type: ${typeof modelName}`);
        
        this.showLoading('Generating forecast...');
    
        try {
            const requestData = {
                forecast_weeks: parseInt(weeks)
            };
            
            if (modelName && modelName.trim() !== '') {
                requestData.model_name = modelName;
                console.log(`🎯 Using specific model: '${modelName}'`);
            } else {
                console.log(`🏆 Using best available model`);
            }
    
            console.log(`📤 Sending request:`, requestData);
    
            const response = await fetch('/api/generate-forecast', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });
    
            const result = await response.json();
            
            console.log(`📥 Received response:`, result);
            
            if (result.success) {
                const actualModel = result.forecast.model_name || modelName || 'best model';
                console.log(`✅ Forecast successful with model: '${actualModel}'`);
                
                this.showAlert(`Forecast generated successfully using ${actualModel} for ${weeks} weeks!`, 'success');
                this.displayForecastResults(result.forecast);
                
                // Force refresh forecast chart immediately
                console.log(`🔄 Triggering chart refresh...`);
                this.forceRefreshForecastChart();
                
                // Also refresh dashboard data
                setTimeout(() => {
                    console.log(`🔄 Triggering dashboard data refresh...`);
                    this.loadDashboardData();
                }, 5000);
                
            } else {
                console.log(`❌ Forecast failed:`, result.message || result.error);
                this.showAlert(`Error: ${result.message || result.error}`, 'danger');
            }
    
        } catch (error) {
            console.error('❌ Forecast error:', error);
            this.showAlert(`Forecast generation failed: ${error.message}`, 'danger');
        } finally {
            console.log("=" .repeat(80));
            this.hideLoading();
        }
    }
        
    async retrainModels() {
        console.log('🔄 Retraining models...');
        
        if (!confirm('Are you sure you want to retrain all models? This may take a few minutes.')) {
            return;
        }

        this.showLoading('Retraining models...');

        try {
            const response = await fetch('/api/retrain-models', {
                method: 'POST'
            });

            const result = await response.json();
            
            if (result.success) {
                this.showAlert('Models retrained successfully!', 'success');
                this.refreshDashboard();
            } else {
                this.showAlert(`Error: ${result.message}`, 'danger');
            }

        } catch (error) {
            console.error('Retrain error:', error);
            this.showAlert(`Retraining failed: ${error.message}`, 'danger');
        } finally {
            this.hideLoading();
        }
    }

    async loadForecastChart() {
        try {
            console.log("📊 LOADING FORECAST CHART:");
            
            const chartDiv = document.getElementById('forecastChart');
            if (!chartDiv) {
                console.log('   ❌ Chart div not found');
                return;
            }

            // Show loading state
            chartDiv.innerHTML = `
                <div class="d-flex justify-content-center align-items-center h-100">
                    <div class="text-center text-muted">
                        <div class="loading-spinner mb-3"></div>
                        <p class="mb-0">Loading forecast chart...</p>
                    </div>
                </div>
            `;

            const timestamp = new Date().getTime();
            const response = await fetch('/api/forecast-chart?' + timestamp);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            
            console.log('   📥 Chart response:', {
                hasError: !!result.error,
                hasData: !!result.data,
                hasLayout: !!result.layout,
                dataLength: result.data ? result.data.length : 0
            });

            if (result.error) {
                chartDiv.innerHTML = `
                    <div class="alert alert-warning m-4">
                        <div class="d-flex align-items-center">
                            <i class="fas fa-exclamation-triangle me-3"></i>
                            <div>
                                <strong>Chart Loading Error</strong>
                                <div class="mt-1">${result.error}</div>
                                <button class="btn btn-sm btn-outline-primary mt-2" onclick="dashboard.loadForecastChart()">
                                    <i class="fas fa-redo me-1"></i>Retry
                                </button>
                            </div>
                        </div>
                    </div>
                `;
                return;
            }

            if (!result.data || !result.layout) {
                chartDiv.innerHTML = `
                    <div class="alert alert-danger m-4">
                        <i class="fas fa-times-circle me-2"></i>
                        Invalid chart data structure
                    </div>
                `;
                return;
            }

            // Clear and create new chart
            chartDiv.innerHTML = '';
            
            try {
                await Plotly.newPlot('forecastChart', result.data, result.layout, result.config || {
                    responsive: true,
                    displayModeBar: true,
                    modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d', 'autoScale2d']
                });
                
                console.log('   ✅ Chart created successfully');
                this.charts.forecast = true;
                
                // Update chart info if available
                const dashboardData = await fetch('/api/data-summary').then(r => r.json());
                if (dashboardData && dashboardData.current_forecast) {
                    this.updateChartInfo(dashboardData.current_forecast);
                }
                
            } catch (plotlyError) {
                console.error('   ❌ Plotly error:', plotlyError);
                chartDiv.innerHTML = `
                    <div class="alert alert-danger m-4">
                        <div class="d-flex align-items-center">
                            <i class="fas fa-chart-line me-3"></i>
                            <div>
                                <strong>Chart Rendering Error</strong>
                                <div class="mt-1">${plotlyError.message}</div>
                                <button class="btn btn-sm btn-outline-primary mt-2" onclick="dashboard.loadForecastChart()">
                                    <i class="fas fa-redo me-1"></i>Try Again
                                </button>
                            </div>
                        </div>
                    </div>
                `;
            }

        } catch (error) {
            console.error('❌ Network error loading forecast chart:', error);
            const chartDiv = document.getElementById('forecastChart');
            if (chartDiv) {
                chartDiv.innerHTML = `
                    <div class="alert alert-danger m-4">
                        <div class="d-flex align-items-center">
                            <i class="fas fa-wifi me-3"></i>
                            <div>
                                <strong>Network Error</strong>
                                <div class="mt-1">Failed to fetch chart data: ${error.message}</div>
                                <button class="btn btn-sm btn-outline-primary mt-2" onclick="dashboard.loadForecastChart()">
                                    <i class="fas fa-redo me-1"></i>Retry
                                </button>
                            </div>
                        </div>
                    </div>
                `;
            }
        }
    }
    
    async loadModelComparisonChart() {
        const container = document.getElementById('modelMetricsContainer');
        if (!container) return;
        
        try {
            // Show loading immediately
            container.innerHTML = `
                <div class="d-flex justify-content-center align-items-center py-4">
                    <div class="spinner-border text-primary me-3" role="status"></div>
                    <span class="text-muted">Loading model metrics...</span>
                </div>
            `;
            
            // Check cache (2 minutes instead of 5)
            const now = Date.now();
            if (modelMetricsCache && (now - modelMetricsCacheTime) < 120000) {
                console.log('Using cached model metrics');
                setTimeout(() => renderModelMetrics(modelMetricsCache), 100);
                return;
            }
            
            console.log('Fetching fresh model metrics...');
            
            // Set timeout for request
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
            
            const response = await fetch('/api/model-comparison-chart', {
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            // Cache the data
            modelMetricsCache = data;
            modelMetricsCacheTime = now;
            
            renderModelMetrics(data);
            
        } catch (error) {
            if (error.name === 'AbortError') {
                console.error('Model metrics request timed out');
                renderModelMetricsError(new Error('Request timed out. Server may be busy processing models.'));
            } else {
                console.error('Error loading model metrics:', error);
                renderModelMetricsError(error);
            }
        }
    }

    renderModelMetrics(data) {
        const container = document.getElementById('modelMetricsContainer');
        if (!container) return;

        if (data.error) {
            renderModelMetricsError(new Error(data.error));
            return;
        }

        const metrics = data.metrics || [];
        let html = '';

        if (metrics.length === 0) {
            html = `
                <div class="text-center text-muted p-4">
                    <i class="fas fa-robot fa-3x mb-3 opacity-25"></i>
                    <p class="mb-2">No trained models found</p>
                    <small>Upload data and train models to see performance metrics</small>
                </div>
            `;
        } else {
            html = '<div class="row g-3">';
            
            metrics.forEach((model, index) => {
                const isFirst = index === 0;
                const borderClass = isFirst ? 'border-success border-2' : '';
                const crownIcon = isFirst ? '<i class="fas fa-crown text-warning position-absolute" style="top: -5px; right: -5px;"></i>' : '';
                
                html += `
                    <div class="col-md-6 col-lg-4">
                        <div class="card h-100 ${borderClass} position-relative" style="transition: transform 0.2s ease;">
                            ${crownIcon}
                            <div class="card-body">
                                <div class="d-flex justify-content-between align-items-start mb-3">
                                    <div>
                                        <h6 class="card-title mb-1 fw-bold">${model.name}</h6>
                                        <small class="text-muted">${model.training_count} trainings</small>
                                    </div>
                                    ${isFirst ? '<span class="badge bg-success">🏆 Best</span>' : ''}
                                </div>
                                
                                <!-- Overall Score Circle -->
                                <div class="text-center mb-3">
                                    <div class="position-relative d-inline-block">
                                        <svg width="80" height="80" class="circular-progress">
                                            <circle cx="40" cy="40" r="35" stroke="#e9ecef" stroke-width="6" fill="none"></circle>
                                            <circle cx="40" cy="40" r="35" 
                                                    stroke="var(--bs-${model.status_color})" 
                                                    stroke-width="6" 
                                                    fill="none"
                                                    stroke-dasharray="220"
                                                    stroke-dashoffset="${220 - (220 * model.overall_score / 100)}"
                                                    stroke-linecap="round"
                                                    transform="rotate(-90 40 40)"
                                                    style="transition: stroke-dashoffset 0.5s ease;">
                                            </circle>
                                        </svg>
                                        <div class="position-absolute top-50 start-50 translate-middle text-center">
                                            <div class="fw-bold text-${model.status_color}">${model.overall_score}%</div>
                                            <small class="text-muted">Score</small>
                                        </div>
                                    </div>
                                </div>
                                
                                <!-- Metrics Grid -->
                                <div class="row g-2 text-center mb-3">
                                    <div class="col-4">
                                        <div class="bg-light rounded p-2">
                                            <div class="fw-bold text-primary small">${model.mae.toFixed(4)}</div>
                                            <div class="text-muted" style="font-size: 0.75rem;">MAE</div>
                                        </div>
                                    </div>
                                    <div class="col-4">
                                        <div class="bg-light rounded p-2">
                                            <div class="fw-bold text-info small">${model.training_time.toFixed(2)}s</div>
                                            <div class="text-muted" style="font-size: 0.75rem;">Speed</div>
                                        </div>
                                    </div>
                                    <div class="col-4">
                                        <div class="bg-light rounded p-2">
                                            <div class="fw-bold text-success small">${model.performance_score.toFixed(1)}%</div>
                                            <div class="text-muted" style="font-size: 0.75rem;">Accuracy</div>
                                        </div>
                                    </div>
                                </div>
                                
                                <!-- Status Badge -->
                                <div class="text-center">
                                    <span class="badge bg-${model.status_color} px-3 py-2">
                                        <i class="fas ${model.status_icon} me-1"></i>
                                        ${model.status.toUpperCase()}
                                    </span>
                                </div>
                                
                                <!-- Trend Indicator -->
                                <div class="mt-2 text-center">
                                    <small class="text-muted">
                                        Trend: 
                                        ${model.trend === 'improving' ? 
                                            '<span class="text-success">📈 Improving</span>' : 
                                            model.trend === 'declining' ? 
                                            '<span class="text-danger">📉 Declining</span>' : 
                                            '<span class="text-secondary">➡️ Stable</span>'
                                        }
                                    </small>
                                </div>
                            </div>
                            
                            <!-- Hover Effect -->
                            <div class="card-footer bg-transparent border-0 p-0">
                                <div class="progress" style="height: 3px;">
                                    <div class="progress-bar bg-${model.status_color}" 
                                        style="width: ${model.overall_score}%"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            });
            
            html += '</div>';
            
            // Add summary stats
            html += `
                <div class="row mt-4">
                    <div class="col-12">
                        <div class="card bg-light border-0">
                            <div class="card-body py-3">
                                <div class="row text-center">
                                    <div class="col-md-3">
                                        <div class="fw-bold text-primary">${data.total_models}</div>
                                        <small class="text-muted">Total Models</small>
                                    </div>
                                    <div class="col-md-3">
                                        <div class="fw-bold text-success">${data.best_model ? data.best_model.name : 'N/A'}</div>
                                        <small class="text-muted">Best Performer</small>
                                    </div>
                                    <div class="col-md-3">
                                        <div class="fw-bold text-info">${data.best_model ? data.best_model.mae.toFixed(4) : 'N/A'}</div>
                                        <small class="text-muted">Best MAE</small>
                                    </div>
                                    <div class="col-md-3">
                                        <div class="fw-bold text-warning">${data.best_model ? data.best_model.overall_score.toFixed(1) : 'N/A'}%</div>
                                        <small class="text-muted">Best Score</small>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        container.innerHTML = html;
        
        // Add hover effects
        container.querySelectorAll('.card').forEach(card => {
            card.addEventListener('mouseenter', function() {
                this.style.transform = 'translateY(-5px)';
                this.style.boxShadow = '0 8px 25px rgba(0,0,0,0.15)';
            });
            
            card.addEventListener('mouseleave', function() {
                this.style.transform = 'translateY(0)';
                this.style.boxShadow = '';
            });
        });
    }

    renderModelMetricsError(error) {
        const container = document.getElementById('modelMetricsContainer');
        if (container) {
            container.innerHTML = `
                <div class="text-center p-5">
                    <div class="mb-4">
                        <i class="fas fa-exclamation-triangle fa-3x text-warning"></i>
                    </div>
                    <h5 class="text-danger mb-3">Error Loading Model Metrics</h5>
                    <p class="text-muted mb-4">${error.message}</p>
                    <div class="d-flex justify-content-center gap-2">
                        <button class="btn btn-outline-primary" onclick="window.dashboard.loadModelComparisonChart()">
                            <i class="fas fa-redo me-1"></i>Retry Loading
                        </button>
                        <button class="btn btn-outline-secondary" onclick="window.dashboard.refreshDashboard()">
                            <i class="fas fa-sync-alt me-1"></i>Refresh Dashboard
                        </button>
                    </div>
                    <div class="mt-3">
                        <small class="text-muted">
                            If the problem persists, try retraining your models or uploading new data.
                        </small>
                    </div>
                </div>
            `;
        }
    }

    async loadDashboardData() {
        try {
            const response = await fetch('/api/data-summary');
            const data = await response.json();
            
            if (data.error) {
                console.error('Dashboard data error:', data.error);
                return;
            }

            this.updateDashboardMetrics(data);
            
            // Load statistical alerts
            await loadStatisticalAlerts();

        } catch (error) {
            console.error('Error loading dashboard data:', error);
        }
    }

    async loadModelPerformanceForModal() {
        try {
            const response = await fetch('/api/model-performance');
            const data = await response.json();
            
            if (data.success && data.models) {
                this.modelPerformanceData = data.models;
                this.updateModelInfo(''); // Update with best model info
            }
        } catch (error) {
            console.error('Error loading model performance:', error);
        }
    }
    
    updateModelInfo(selectedModel) {
        const modelInfoDiv = document.getElementById('modelInfo');
        
        if (!this.modelPerformanceData) {
            modelInfoDiv.innerHTML = '<small class="text-muted">Model performance data not available</small>';
            return;
        }
    
        let infoHTML = '';
        
        if (!selectedModel || selectedModel === '') {
            // Show best model info
            let bestModel = null;
            let bestMAE = Infinity;
            
            for (const [modelName, performance] of Object.entries(this.modelPerformanceData)) {
                if (performance.latest_mae < bestMAE) {
                    bestMAE = performance.latest_mae;
                    bestModel = { name: modelName, ...performance };
                }
            }
            
            if (bestModel) {
                infoHTML = `
                    <div class="row text-center">
                        <div class="col-4">
                            <strong class="text-primary">${bestModel.name}</strong>
                            <br><small class="text-muted">Best Model</small>
                        </div>
                        <div class="col-4">
                            <strong class="text-success">${bestModel.latest_mae.toFixed(4)}</strong>
                            <br><small class="text-muted">MAE</small>
                        </div>
                        <div class="col-4">
                            <strong class="text-info">${bestModel.training_count}</strong>
                            <br><small class="text-muted">Trainings</small>
                        </div>
                    </div>
                `;
            }
        } else {
            // Show selected model info
            const modelData = this.modelPerformanceData[selectedModel];
            if (modelData) {
                const trendIcon = modelData.trend_direction === 'improving' ? '📈' : 
                                 modelData.trend_direction === 'declining' ? '📉' : '➡️';
                
                infoHTML = `
                    <div class="row text-center mb-2">
                        <div class="col-3">
                            <strong class="text-primary">${modelData.latest_mae.toFixed(4)}</strong>
                            <br><small class="text-muted">Latest MAE</small>
                        </div>
                        <div class="col-3">
                            <strong class="text-success">${modelData.best_mae.toFixed(4)}</strong>
                            <br><small class="text-muted">Best MAE</small>
                        </div>
                        <div class="col-3">
                            <strong class="text-info">${modelData.training_count}</strong>
                            <br><small class="text-muted">Trainings</small>
                        </div>
                        <div class="col-3">
                            <strong class="text-warning">${trendIcon}</strong>
                            <br><small class="text-muted">${modelData.trend_direction}</small>
                        </div>
                    </div>
                    <div class="text-center">
                        <small class="text-muted">
                            Avg training time: ${modelData.avg_training_time.toFixed(3)}s
                        </small>
                    </div>
                `;
            }
        }
        
        modelInfoDiv.innerHTML = infoHTML;
    }
    
    updateDashboardMetrics(data) {
        // Update metric cards
        const totalRecords = document.getElementById('totalRecords');
        if (totalRecords) {
            totalRecords.textContent = data.total_records || 0;
        }

        const latestValue = document.getElementById('latestValue');
        if (latestValue && data.latest_value !== null) {
            latestValue.textContent = `${data.latest_value.toFixed(3)}%`;
        }

        const dataRange = document.getElementById('dataRange');
        if (dataRange && data.date_range) {
            dataRange.textContent = `${data.date_range.start} to ${data.date_range.end}`;
        }

        // Update data quality indicator
        const qualityScore = document.getElementById('qualityScore');
        if (qualityScore && data.data_quality) {
            const score = data.data_quality.completeness_percent;
            qualityScore.textContent = `${score.toFixed(1)}%`;
            
            // Update quality indicator color
            const indicator = qualityScore.parentElement?.querySelector('.status-indicator');
            if (indicator) {
                indicator.className = 'status-indicator ';
                if (score >= 90) indicator.className += 'status-success';
                else if (score >= 70) indicator.className += 'status-warning';
                else indicator.className += 'status-danger';
            }
        }
    }

    displayUploadResults(result) {
        const resultsDiv = document.getElementById('uploadResults');
        if (!resultsDiv) return;

        const html = `
            <div class="card mt-3 fade-in">
                <div class="card-header bg-success text-white">
                    <h6 class="mb-0"><i class="fas fa-check-circle me-2"></i>Upload Results</h6>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-6">
                            <h6>Data Processing</h6>
                            <ul class="list-unstyled">
                                <li><i class="fas fa-database text-primary me-2"></i>Total Records: ${result.data_processing.merge_info.total_records}</li>
                                <li><i class="fas fa-plus text-success me-2"></i>New Records: ${result.data_processing.merge_info.new_records}</li>
                                ${result.data_processing.merge_info.duplicates_removed > 0 ? 
                                    `<li><i class="fas fa-minus text-warning me-2"></i>Duplicates Removed: ${result.data_processing.merge_info.duplicates_removed}</li>` : ''}
                            </ul>
                        </div>
                        <div class="col-md-6">
                            <h6>Best Model</h6>
                            <ul class="list-unstyled">
                                <li><i class="fas fa-trophy text-warning me-2"></i>Model: ${result.best_model.name}</li>
                                <li><i class="fas fa-chart-line text-info me-2"></i>MAE: ${result.best_model.performance.mae.toFixed(4)}</li>
                                <li><i class="fas fa-arrow-up text-success me-2"></i>Improvement: ${result.best_model.is_improvement ? 'Yes' : 'No'}</li>
                            </ul>
                        </div>
                    </div>
                    <div class="mt-3">
                        <h6>Forecast Summary</h6>
                        <p class="mb-0">
                            <i class="fas fa-calendar me-2"></i>${result.forecast.weeks_forecasted} weeks forecast generated
                            <span class="badge bg-primary ms-2">${result.forecast.summary.avg_prediction.toFixed(3)}% avg</span>
                            <span class="badge ${result.forecast.summary.trend === 'Naik' ? 'bg-success' : 'bg-danger'} ms-2">
                                ${result.forecast.summary.trend === 'Naik' ? '↗' : '↘'} ${result.forecast.summary.trend}
                            </span>
                        </p>
                    </div>
                </div>
            </div>
        `;

        resultsDiv.innerHTML = html;
        
        // Auto-hide after 10 seconds
        setTimeout(() => {
            resultsDiv.innerHTML = '';
        }, 10000);
    }

    displayForecastResults(forecast) {
        const resultsDiv = document.getElementById('forecastResults');
        if (!resultsDiv) return;

        const html = `
            <div class="card mt-3 fade-in">
                <div class="card-header bg-info text-white">
                    <h6 class="mb-0"><i class="fas fa-chart-line me-2"></i>Forecast Results</h6>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-4">
                            <div class="text-center">
                                <h4 class="text-primary">${forecast.summary.avg_prediction.toFixed(3)}%</h4>
                                <small class="text-muted">Average Prediction</small>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="text-center">
                                <h4 class="${forecast.summary.trend === 'Naik' ? 'text-success' : 'text-danger'}">
                                    ${forecast.summary.trend === 'Naik' ? '↗' : '↘'} ${forecast.summary.trend}
                                </h4>
                                <small class="text-muted">Trend Direction</small>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="text-center">
                                <h4 class="text-info">${forecast.weeks_forecasted}</h4>
                                <small class="text-muted">Weeks Forecasted</small>
                            </div>
                        </div>
                    </div>
                    <div class="mt-3">
                        <small class="text-muted">
                            <i class="fas fa-robot me-1"></i>Model: ${forecast.model_name} | 
                            <i class="fas fa-clock me-1"></i>Generated: ${new Date().toLocaleString()}
                        </small>
                    </div>
                </div>
            </div>
        `;

        resultsDiv.innerHTML = html;
    }

    showForecastModal() {
        // Create modal HTML with dynamic model options
        const modalHTML = `
            <div class="modal fade" id="forecastModal" tabindex="-1">
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header bg-primary text-white">
                            <h5 class="modal-title">
                                <i class="fas fa-chart-line me-2"></i>Generate Forecast
                            </h5>
                            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <form id="forecastForm">
                                <div class="row">
                                    <div class="col-md-6">
                                        <div class="mb-3">
                                            <label for="forecastWeeksModal" class="form-label">
                                                <i class="fas fa-calendar-alt me-1 text-primary"></i>
                                                <strong>Forecast Weeks</strong>
                                            </label>
                                            <select class="form-select form-select-lg" id="forecastWeeksModal" required>
                                                <option value="4">4 weeks</option>
                                                <option value="6">6 weeks</option>
                                                <option value="8" selected>8 weeks</option>
                                                <option value="10">10 weeks</option>
                                                <option value="12">12 weeks</option>
                                            </select>
                                            <div class="form-text">
                                                <i class="fas fa-info-circle me-1"></i>
                                                Number of weeks to forecast into the future
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-md-6">
                                        <div class="mb-3">
                                            <label for="modelSelect" class="form-label">
                                                <i class="fas fa-robot me-1 text-success"></i>
                                                <strong>Model Selection</strong>
                                            </label>
                                            <select class="form-select form-select-lg" id="modelSelect">
                                                <option value="">🏆 Use best model (Recommended)</option>
                                                <option value="XGBoost_Advanced">🚀 XGBoost Advanced</option>
                                                <option value="Random_Forest">🌳 Random Forest</option>
                                                <option value="LightGBM">⚡ LightGBM</option>
                                                <option value="KNN">📍 K-Nearest Neighbors</option>
                                            </select>
                                            <div class="form-text">
                                                <i class="fas fa-lightbulb me-1"></i>
                                                Leave empty to automatically use the best performing model
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="mb-3">
                                    <div class="card bg-light border-info">
                                        <div class="card-header bg-info text-white">
                                            <h6 class="card-title mb-0">
                                                <i class="fas fa-info-circle me-1"></i>
                                                Model Performance Information
                                            </h6>
                                        </div>
                                        <div class="card-body p-3">
                                            <div id="modelInfo">
                                                <div class="text-center text-muted">
                                                    <div class="spinner-border spinner-border-sm me-2" role="status"></div>
                                                    Loading model performance data...
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </form>
                        </div>
                        <div class="modal-footer bg-light">
                            <button type="button" class="btn btn-secondary btn-lg" data-bs-dismiss="modal">
                                <i class="fas fa-times me-1"></i>Cancel
                            </button>
                            <button type="button" class="btn btn-primary btn-lg" id="generateForecastBtn" onclick="dashboard.submitForecastForm()">
                                <i class="fas fa-chart-line me-1"></i>Generate Forecast
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    
        // Remove existing modal if any
        const existingModal = document.getElementById('forecastModal');
        if (existingModal) {
            existingModal.remove();
        }
    
        // Add modal to page
        document.body.insertAdjacentHTML('beforeend', modalHTML);
    
        // Load model performance data
        this.loadModelPerformanceForModal();
    
        // Setup model selection change handler
        document.getElementById('modelSelect').addEventListener('change', (e) => {
            this.updateModelInfo(e.target.value);
        });
    
        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('forecastModal'));
        modal.show();
    }

    submitForecastForm() {
        const weeks = document.getElementById('forecastWeeksModal').value;
        const model = document.getElementById('modelSelect').value;
    
        console.log(`🔮 Generating forecast: ${weeks} weeks, model: ${model || 'best'}`);
    
        // Hide modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('forecastModal'));
        modal.hide();
    
        // Generate forecast with selected parameters
        this.generateForecast(model || null, parseInt(weeks));
    }
    
    refreshCharts() {
        console.log('🔄 Refreshing charts...');
        this.loadForecastChart();
        this.loadModelComparisonChart();
    }

    refreshForecastChart() {
        console.log('🔄 Manual refresh forecast chart...');
        
        if (window.dashboard) {
            // Show loading on button
            const refreshBtn = document.getElementById('refreshChartBtn');
            if (refreshBtn) {
                refreshBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Refreshing...';
                refreshBtn.disabled = true;
            }
            
            window.dashboard.forceRefreshForecastChart();
            window.dashboard.showAlert('Forecast chart is being refreshed...', 'info');
            
            // Reset button after delay
            setTimeout(() => {
                if (refreshBtn) {
                    refreshBtn.innerHTML = '<i class="fas fa-sync-alt me-1"></i>Refresh Chart';
                    refreshBtn.disabled = false;
                }
            }, 3000);
        }
    }
    refreshChartsAfterForecast() {
        console.log('🔄 Refreshing charts after forecast generation...');
        
        // Clear existing chart first
        const chartDiv = document.getElementById('forecastChart');
        if (chartDiv) {
            chartDiv.innerHTML = `
                <div class="d-flex justify-content-center align-items-center h-100">
                    <div class="text-center text-muted">
                        <div class="loading-spinner mb-3"></div>
                        <p class="mb-0">Refreshing forecast chart...</p>
                    </div>
                </div>
            `;
        }
        
        // Add delay to ensure data is saved
        setTimeout(() => {
            this.loadForecastChart();
            this.loadModelComparisonChart();
        }, 2000);
    }

    forceRefreshForecastChart() {
        console.log('🔄 Force refreshing forecast chart...');
        
        const chartDiv = document.getElementById('forecastChart');
        if (chartDiv) {
            // Show loading state
            chartDiv.innerHTML = `
                <div class="d-flex justify-content-center align-items-center h-100">
                    <div class="text-center text-muted">
                        <div class="loading-spinner mb-3"></div>
                        <p class="mb-0">Updating forecast chart with new model...</p>
                    </div>
                </div>
            `;
            
            // Clear any existing chart data
            if (this.charts.forecast) {
                try {
                    Plotly.purge('forecastChart');
                } catch (e) {
                    console.log('Chart purge not needed');
                }
            }
            
            // 🚨 PERBAIKAN: Refresh immediately karena data sudah di-save di memory
            setTimeout(() => {
                this.loadForecastChart();
            }, 1000); // Reduced to 1 second
        }
    }    
    
    refreshDashboard() {
        console.log('🔄 Refreshing dashboard...');
        this.loadDashboardData();
        this.refreshCharts();
        this.showAlert('Dashboard refreshed successfully!', 'info');
    }

    setupAutoRefresh() {
        // Auto-refresh every 5 minutes
        this.refreshInterval = setInterval(() => {
            console.log('⏰ Auto-refreshing dashboard...');
            this.loadDashboardData();
        }, 5 * 60 * 1000);
    }

    showLoading(message = 'Loading...') {
        // Create loading overlay
        const loadingHTML = `
            <div id="loadingOverlay" class="position-fixed top-0 start-0 w-100 h-100 d-flex align-items-center justify-content-center" 
                 style="background-color: rgba(0,0,0,0.7); z-index: 9999;">
                <div class="text-center text-white">
                    <div class="loading-spinner mb-3" style="width: 3rem; height: 3rem; border-width: 4px;"></div>
                    <h5>${message}</h5>
                </div>
            </div>
        `;

        // Remove existing overlay
        const existingOverlay = document.getElementById('loadingOverlay');
        if (existingOverlay) {
            existingOverlay.remove();
        }

        document.body.insertAdjacentHTML('beforeend', loadingHTML);
    }

    hideLoading() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.remove();
        }
    }

    showAlert(message, type = 'info') {
        const alertHTML = `
            <div class="alert alert-${type} alert-dismissible fade show position-fixed" 
                 style="top: 20px; right: 20px; z-index: 9999; max-width: 400px;" role="alert">
                <i class="fas fa-${this.getAlertIcon(type)} me-2"></i>
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;

        document.body.insertAdjacentHTML('beforeend', alertHTML);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            const alerts = document.querySelectorAll('.alert');
            alerts.forEach(alert => {
                if (alert.textContent.includes(message.substring(0, 20))) {
                    alert.remove();
                }
            });
        }, 5000);
    }

    updateRealTimeStatus() {
        const lastUpdateElement = document.getElementById('lastUpdate');
        const dataFreshnessElement = document.getElementById('dataFreshness');
        
        if (lastUpdateElement) {
            const now = new Date();
            lastUpdateElement.textContent = now.toLocaleTimeString();
        }
        
        if (dataFreshnessElement) {
            // Simulate data freshness check
            const freshnessClasses = ['bg-success', 'bg-warning', 'bg-danger'];
            const freshnessTexts = ['Fresh', 'Stale', 'Old'];
            const randomIndex = Math.floor(Math.random() * 3);
            
            dataFreshnessElement.className = `badge ${freshnessClasses[randomIndex]}`;
            dataFreshnessElement.textContent = freshnessTexts[randomIndex];
        }
    }    

    getAlertIcon(type) {
        const icons = {
            'success': 'check-circle',
            'danger': 'exclamation-triangle',
            'warning': 'exclamation-circle',
            'info': 'info-circle'
        };
        return icons[type] || 'info-circle';
    }

    async exportData(dataType = 'all') {
        console.log(`📤 Exporting ${dataType} data...`);
        
        this.showLoading(`Preparing ${dataType} data for export...`);
        
        try {
            const response = await fetch(`/api/export-data?type=${dataType}`);
            const result = await response.json();
            
            if (result.success) {
                // Show export success info
                this.showAlert(`Export successful! ${result.records} records exported.`, 'success');
                
                // Trigger download
                const downloadLink = document.createElement('a');
                downloadLink.href = result.download_url;
                downloadLink.download = result.filename;
                document.body.appendChild(downloadLink);
                downloadLink.click();
                document.body.removeChild(downloadLink);
                
                console.log(`✅ Export completed: ${result.filename}`);
                
            } else {
                this.showAlert(`Export failed: ${result.message}`, 'danger');
            }
            
        } catch (error) {
            console.error('Export error:', error);
            this.showAlert(`Export failed: ${error.message}`, 'danger');
        } finally {
            this.hideLoading();
        }
    }

    showExportModal() {
        const modalHTML = `
            <div class="modal fade" id="exportModal" tabindex="-1">
                <div class="modal-dialog">
                    <div class="modal-content">
                        <div class="modal-header bg-success text-white">
                            <h5 class="modal-title">
                                <i class="fas fa-file-csv me-2"></i>Export Data to CSV
                            </h5>
                            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <p class="text-muted mb-4">Choose what data you want to export:</p>
                            
                            <div class="list-group">
                                <button type="button" class="list-group-item list-group-item-action d-flex align-items-center" 
                                        onclick="dashboard.exportData('historical'); bootstrap.Modal.getInstance(document.getElementById('exportModal')).hide();">
                                    <div class="me-3">
                                        <i class="fas fa-database fa-2x text-primary"></i>
                                    </div>
                                    <div>
                                        <h6 class="mb-1">Historical Data</h6>
                                        <small class="text-muted">Export all historical IPH data</small>
                                    </div>
                                </button>
                                
                                <button type="button" class="list-group-item list-group-item-action d-flex align-items-center mt-2" 
                                        onclick="dashboard.exportData('forecast'); bootstrap.Modal.getInstance(document.getElementById('exportModal')).hide();">
                                    <div class="me-3">
                                        <i class="fas fa-chart-line fa-2x text-info"></i>
                                    </div>
                                    <div>
                                        <h6 class="mb-1">Forecast Data</h6>
                                        <small class="text-muted">Export current forecast predictions</small>
                                    </div>
                                </button>
                                
                                <button type="button" class="list-group-item list-group-item-action d-flex align-items-center mt-2" 
                                        onclick="dashboard.exportData('all'); bootstrap.Modal.getInstance(document.getElementById('exportModal')).hide();">
                                    <div class="me-3">
                                        <i class="fas fa-file-export fa-2x text-success"></i>
                                    </div>
                                    <div>
                                        <h6 class="mb-1">Complete Dataset</h6>
                                        <small class="text-muted">Export historical + forecast data combined</small>
                                    </div>
                                </button>
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">
                                <i class="fas fa-times me-1"></i>Cancel
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Remove existing modal
        const existingModal = document.getElementById('exportModal');
        if (existingModal) {
            existingModal.remove();
        }
        
        // Add modal to page
        document.body.insertAdjacentHTML('beforeend', modalHTML);
        
        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('exportModal'));
        modal.show();
    }

    updateChartInfo(forecast) {
        console.log('🔄 Updating chart info badges:', forecast);
        
        // Update model name badge
        const modelNameSpan = document.getElementById('currentModelName');
        if (modelNameSpan && forecast) {
            const modelName = forecast.model_name ? 
                forecast.model_name.replace('_', ' ') : 'Unknown Model';
            modelNameSpan.textContent = modelName;
            console.log(`   🤖 Updated model name: ${modelName}`);
        }
        
        // Update weeks badge  
        const weeksCountSpan = document.getElementById('currentWeeksCount');
        if (weeksCountSpan && forecast) {
            const weeks = forecast.weeks_forecasted || 0;
            weeksCountSpan.textContent = weeks;
            console.log(`   📅 Updated weeks count: ${weeks}`);
        }
        
        // Update refresh button state temporarily
        const refreshBtn = document.getElementById('refreshChartBtn');
        if (refreshBtn) {
            const originalHTML = refreshBtn.innerHTML;
            refreshBtn.innerHTML = '<i class="fas fa-check me-1"></i>Updated';
            refreshBtn.classList.remove('btn-outline-primary');
            refreshBtn.classList.add('btn-success');
            
            setTimeout(() => {
                refreshBtn.innerHTML = originalHTML;
                refreshBtn.classList.remove('btn-success');
                refreshBtn.classList.add('btn-outline-primary');
            }, 2000);
        }
    }
    
}

async function quickRetrain() {
    if (confirm('Retrain all models? This will take a few minutes.')) {
        showLoading('Retraining models...');
        try {
            const response = await fetch('/api/retrain-models', { method: 'POST' });
            const result = await response.json();
            
            if (result.success) {
                showAlert('Models retrained successfully!', 'success');
                updateDataControlMetrics();
            } else {
                showAlert('Retrain failed: ' + result.message, 'danger');
            }
        } catch (error) {
            showAlert('Error: ' + error.message, 'danger');
        } finally {
            hideLoading();
        }
    }
}

async function validateData() {
    showLoading('Validating data quality...');
    try {
        const response = await fetch('/api/validate-data');
        const result = await response.json();
        
        if (result.success) {
            showAlert(`Data validation complete. Quality score: ${result.quality_score}%`, 'info');
            updateDataQualityIndicators(result);
        }
    } catch (error) {
        showAlert('Validation failed: ' + error.message, 'danger');
    } finally {
        hideLoading();
    }
}

async function createBackup() {
    showLoading('Creating backup...');
    try {
        const response = await fetch('/api/create-backup', { method: 'POST' });
        const result = await response.json();
        
        if (result.success) {
            showAlert('Backup created successfully!', 'success');
        }
    } catch (error) {
        showAlert('Backup failed: ' + error.message, 'danger');
    } finally {
        hideLoading();
    }
}

async function loadStatisticalAlerts() {
    try {
        const response = await fetch('/api/alerts/statistical');
        const data = await response.json();
        
        if (!data.success || !data.alerts || data.alerts.length === 0) {
            document.getElementById('statisticalAlertsPanel').style.display = 'none';
            return;
        }
        
        const panel = document.getElementById('statisticalAlertsPanel');
        let alertsHtml = '';
        
        data.alerts.forEach(alert => {
            const alertClass = {
                'critical': 'alert-danger',
                'warning': 'alert-warning', 
                'info': 'alert-info'
            }[alert.type] || 'alert-info';
            
            const icon = {
                'critical': 'fa-exclamation-triangle',
                'warning': 'fa-exclamation-circle',
                'info': 'fa-info-circle'
            }[alert.type] || 'fa-info-circle';
            
            alertsHtml += `
                <div class="alert ${alertClass} border-0 shadow-sm mb-3" role="alert">
                    <div class="d-flex align-items-center">
                        <i class="fas ${icon} fa-lg me-3"></i>
                        <div class="flex-grow-1">
                            <strong>${alert.title}</strong>
                            <div class="mt-1">${alert.message}</div>
                            <small class="text-muted">
                                <i class="fas fa-calendar me-1"></i>
                                ${alert.date}
                            </small>
                        </div>
                        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                    </div>
                </div>
            `;
        });
        
        panel.innerHTML = alertsHtml;
        panel.style.display = 'block';
        
    } catch (error) {
        console.error('Error loading statistical alerts:', error);
    }
}

function updateDataControlMetrics() {
    // Update semua metrics di panel
    // Call API untuk get latest status
    fetch('/api/data-control-status')
        .then(response => response.json())
        .then(data => {
            document.getElementById('dataHealthScore').textContent = data.health_score + '%';
            document.getElementById('modelStatus').textContent = data.model_status;
            document.getElementById('processingSpeed').textContent = data.processing_speed;
            document.getElementById('forecastAccuracy').textContent = data.forecast_accuracy + '%';
        });
}

document.addEventListener('DOMContentLoaded', function() {
    window.dashboard = new ForecastingDashboard();
    
    // Load charts if chart containers exist
    if (document.getElementById('forecastChart')) {
        dashboard.loadForecastChart();
    }
    
    if (document.getElementById('modelComparisonChart')) {
        dashboard.loadModelComparisonChart();
    }
});

// Export for global access
window.ForecastingDashboard = ForecastingDashboard;