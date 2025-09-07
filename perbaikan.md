## 📍 **Lokasi Spesifik untuk Kode Perbaikan Visual Chart**

### **1. Chart Forecast - Style Improvements**

**A. Gradient & Smooth Curves:**

- File: `app.py`
- Function: `forecast_chart()`
- Line: ~580-620 (bagian `fig.add_trace` untuk historical data)

**B. Interactive Confidence Band:**

- File: `app.py` 
- Function: `forecast_chart()`
- Line: ~650-680 (bagian confidence interval traces)

**C. Modern Layout & Styling:**

- File: `app.py`
- Function: `forecast_chart()`
- Line: ~700-750 (bagian `fig.update_layout`)

### **2. Visualisasi Data - Modern Styling**

**A. Dekomposisi Time Series:**

- File: `services/visualization_service.py`
- Function: `perform_decomposition()`
- Line: ~60-120 (bagian create traces dan layout)

**B. Moving Averages - Neon Glow:**

- File: `services/visualization_service.py`
- Function: `calculate_moving_averages()`
- Line: ~150-200 (bagian add traces)

**C. Volatility - Area Chart Pattern:**

- File: `services/visualization_service.py`
- Function: `analyze_volatility()`
- Line: ~250-300 (bagian create figure dan traces)

### **3. Interactive Elements**

**A. Hover Effects:**

- File: `app.py`
- Function: `forecast_chart()`
- Line: ~590, 630 (bagian `hovertemplate` di setiap trace)

**B. Animation & Transitions:**

- File: `static/js/dashboard.js`
- Function: `loadForecastChart()`
- Line: ~180-200 (bagian `Plotly.newPlot`)

### **4. Color Schemes Modern**

**A. Gradient Color Palettes:**

- File: `app.py`
- Function: `forecast_chart()`
- Line: ~560-570 (bagian model_colors dictionary)

**B. Dark Mode Support:**

- File: `static/css/style.css`
- Location: Tambah di akhir file (~line 300+)

### **5. Responsive & Mobile-Friendly**

**A. Adaptive Layout:**

- File: `static/js/dashboard.js`
- Function: `loadForecastChart()`
- Line: ~200-220 (setelah Plotly.newPlot)

**B. Touch-Friendly Controls:**

- File: `static/js/dashboard.js`
- Function: `loadForecastChart()`
- Line: ~200-220 (dalam config object)

### **6. Micro-Interactions**

**A. Loading Animations:**

- File: `static/css/style.css`
- Location: Line ~250-300 (bagian loading spinner)

**B. Smooth State Transitions:**

- File: `static/js/dashboard.js`
- Function: `forceRefreshForecastChart()`
- Line: ~350-370 (bagian chart refresh)