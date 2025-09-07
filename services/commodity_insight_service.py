
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import re
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class CommodityInsightService:
    """Service for analyzing commodity impacts and generating insights"""
    
    def __init__(self, commodity_data_path='data/IPH-Kota-Batu.csv'):
        self.commodity_data_path = commodity_data_path
        self.commodity_cache = None
        self.last_cache_time = None
        self.cache_duration = 300  # 5 minutes cache
        
    def load_commodity_data(self):
        """Load commodity data from CSV"""
        try:
            if not os.path.exists(self.commodity_data_path):
                print(f"⚠️ Commodity data not found at {self.commodity_data_path}")
                return pd.DataFrame()
            
            df = pd.read_csv(self.commodity_data_path)
            df.columns = df.columns.str.strip()  # strip spaces

            # Rename columns for consistency (mapping real from IPH Kota Batu.csv)
            column_mapping = {
                ' Indikator Perubahan Harga (%)': 'IPH',
                'Komoditas Andil Perubahan Harga ': 'Komoditas_Andil',
                'Komoditas Fluktuasi Harga Tertinggi': 'Komoditas_Fluktuasi_Tertinggi',
                'Fluktuasi Harga': 'Nilai_Fluktuasi',
                'Minggu ke-': 'Minggu',
                'Kab/Kota': 'Kota',
                'Bulan': 'Bulan',
            }
            for old, new in column_mapping.items():
                if old in df.columns:
                    df.rename(columns={old: new}, inplace=True)

            # Isi forward kolom Bulan yang kosong
            if 'Bulan' in df.columns:
                df['Bulan'] = df['Bulan'].fillna(method='ffill')
            if 'Minggu' in df.columns:
                df['Minggu'] = df['Minggu'].fillna(method='ffill')
            if 'Kota' in df.columns:
                df['Kota'] = df['Kota'].fillna(method='ffill')

            # Convert IPH dan Nilai_Fluktuasi ke numeric
            if 'IPH' in df.columns:
                df['IPH'] = pd.to_numeric(df['IPH'], errors='coerce')
            if 'Nilai_Fluktuasi' in df.columns:
                df['Nilai_Fluktuasi'] = pd.to_numeric(df['Nilai_Fluktuasi'], errors='coerce')

            df['Tanggal'] = [self.to_weekly_date(b, m) for b, m in zip(df['Bulan'], df['Minggu'])]

            print(f"✅ Loaded {len(df)} commodity records")
            return df
        except Exception as e:
            print(f"❌ Error loading commodity data: {str(e)}")
            return pd.DataFrame()
        
    def to_weekly_date(self, bulan, minggu, tahun_default=2023):
        """
        Mengubah Bulan + Minggu ke- menjadi pd.Timestamp (awal minggu)
        Handle 'Januari', 'Februari', ... serta format tahun seperti 'Januari '24'
        """
        import calendar
        # Parsing tahun
        bulan = str(bulan).strip()
        if "'" in bulan:
            parts = bulan.split("'")
            month_str = parts[0].strip()
            year = 2000 + int(parts[1])
        elif " " in bulan and bulan.split(" ")[-1].isdigit():
            # Format: Maret 2024
            parts = bulan.split(" ")
            month_str = " ".join(parts[:-1])
            year = int(parts[-1])
        else:
            month_str = bulan
            year = tahun_default

        # Bulan ke int
        month_map = {
            'januari':1, 'februari':2, 'maret':3, 'april':4, 'mei':5, 'juni':6,
            'juli':7, 'agustus':8, 'september':9, 'oktober':10, 'november':11, 'desember':12
        }
        month_key = month_str.strip().lower().replace("'", "").replace(' ', '')
        for k in month_map:
            if month_key.startswith(k):
                month = month_map[k]
                break
        else:
            month = 1

        # Minggu ke int
        if isinstance(minggu, str) and minggu.strip().upper().startswith('M'):
            week = int("".join(filter(str.isdigit, minggu)))
        else:
            try:
                week = int(minggu)
            except:
                week = 1

        # Tanggal awal minggu (misal M1 = tanggal 1)
        day = 1 + (week - 1) * 7
        try:
            return pd.Timestamp(year, month, day)
        except:
            return pd.NaT      
                 
    def parse_commodity_impacts(self, commodity_string):
        """Parse commodity impact string into structured data"""
        if pd.isna(commodity_string) or not commodity_string:
            return []
        commodities = []

        # Ganti koma pada angka dengan titik (misal -0,773 -> -0.773)
        # dan hapus spasi di depan dan belakang
        s = str(commodity_string).replace(',', '.').replace(';', '; ')
        # Pattern: NAMA(angka)
        import re
        # Support juga string seperti: "CABAI RAWIT(-1.0273), CABAI MERAH(-0.9684), DAGING AYAM RAS(-0.0618)"
        pattern = r'([A-Z\s/]+)\((-?\d+\.?\d*)\)'
        matches = re.findall(pattern, s.upper())

        for match in matches:
            commodity_name = match[0].strip()
            impact_value = float(match[1])
            commodities.append({
                'name': commodity_name,
                'impact': impact_value
            })
        return commodities
    
    def get_current_week_insights(self):
        """Get insights for current week"""
        df = self.load_commodity_data()
        
        if df.empty:
            return {
                'success': False,
                'message': 'No commodity data available'
            }
        
        # Get latest data
        latest_record = df.iloc[-1] if not df.empty else None
        
        if latest_record is None:
            return {
                'success': False,
                'message': 'No recent data found'
            }
        
        # Parse commodity impacts
        commodities = self.parse_commodity_impacts(latest_record.get('Komoditas_Andil', ''))
        
        # Sort by absolute impact
        commodities.sort(key=lambda x: abs(x['impact']), reverse=True)
        
        # Categorize impacts
        positive_impacts = [c for c in commodities if c['impact'] > 0]
        negative_impacts = [c for c in commodities if c['impact'] < 0]
        
        insights = {
            'success': True,
            'period': {
                'bulan': latest_record.get('Bulan', 'Unknown'),
                'minggu': latest_record.get('Minggu', 'Unknown'),
                'kota': latest_record.get('Kab/Kota', 'Unknown')
            },
            'iph_value': float(latest_record.get('IPH', 0)),
            'top_positive_impacts': positive_impacts[:3],
            'top_negative_impacts': negative_impacts[:3],
            'most_volatile': latest_record.get('Komoditas_Fluktuasi_Tertinggi', 'N/A'),
            'volatility_value': float(latest_record.get('Nilai_Fluktuasi', 0)),
            'summary': self._generate_weekly_summary(latest_record, commodities)
        }
        
        return insights
    
    def _generate_weekly_summary(self, record, commodities):
        """Generate human-readable summary"""
        iph = record.get('IPH', 0)
        bulan = record.get('Bulan', 'Unknown')
        minggu = record.get('Minggu', 'Unknown')
        
        if iph > 0:
            trend = "inflasi"
            icon = "📈"
        elif iph < 0:
            trend = "deflasi"
            icon = "📉"
        else:
            trend = "stabil"
            icon = "➡️"
        
        # Get top commodity
        if commodities:
            top_commodity = commodities[0]
            impact_text = f"{top_commodity['name']} ({top_commodity['impact']:.3f}%)"
        else:
            impact_text = "Tidak ada data komoditas"
        
        summary = f"{icon} {bulan} {minggu}: Terjadi {trend} sebesar {abs(iph):.2f}%. "
        summary += f"Komoditas paling berpengaruh: {impact_text}. "
        
        if record.get('Komoditas_Fluktuasi_Tertinggi'):
            summary += f"Komoditas paling volatile: {record['Komoditas_Fluktuasi_Tertinggi']} "
            summary += f"dengan fluktuasi {record.get('Nilai_Fluktuasi', 0):.2%}."
        
        return summary
    
    def get_monthly_analysis(self, month=None):
        """Analyze commodity patterns for a specific month"""
        df = self.load_commodity_data()
        
        if df.empty:
            return {'success': False, 'message': 'No data available'}
        
        # Filter by month if specified
        if month:
            df_month = df[df['Bulan'].str.lower() == month.lower()]
        else:
            # Get latest month
            df_month = df[df['Bulan'] == df.iloc[-1]['Bulan']]
        
        if df_month.empty:
            return {'success': False, 'message': f'No data for month: {month}'}
        
        # Aggregate commodity impacts
        all_commodities = defaultdict(list)
        
        for _, row in df_month.iterrows():
            commodities = self.parse_commodity_impacts(row.get('Komoditas_Andil', ''))
            for comm in commodities:
                all_commodities[comm['name']].append(comm['impact'])
        
        # Calculate statistics
        commodity_stats = []
        for name, impacts in all_commodities.items():
            commodity_stats.append({
                'name': name,
                'avg_impact': np.mean(impacts),
                'total_impact': sum(impacts),
                'frequency': len(impacts),
                'max_impact': max(impacts),
                'min_impact': min(impacts),
                'volatility': np.std(impacts) if len(impacts) > 1 else 0
            })
        
        # Sort by total impact
        commodity_stats.sort(key=lambda x: abs(x['total_impact']), reverse=True)
        
        # Get volatility leaders
        volatility_commodities = df_month['Komoditas_Fluktuasi_Tertinggi'].value_counts().to_dict()
        
        return {
            'success': True,
            'month': df_month.iloc[0]['Bulan'],
            'weeks_analyzed': len(df_month),
            'avg_iph': float(df_month['IPH'].mean()),
            'iph_trend': 'Inflasi' if df_month['IPH'].mean() > 0 else 'Deflasi',
            'top_impact_commodities': commodity_stats[:5],
            'most_volatile_commodities': [
                {'name': k, 'frequency': v} 
                for k, v in list(volatility_commodities.items())[:3]
            ],
            'monthly_summary': self._generate_monthly_summary(df_month, commodity_stats)
        }
    
    def _generate_monthly_summary(self, df_month, commodity_stats):
        """Generate monthly summary text"""
        month = df_month.iloc[0]['Bulan']
        avg_iph = df_month['IPH'].mean()
        
        summary = f"📊 Analisis {month}:\n"
        summary += f"• Rata-rata IPH: {avg_iph:.2f}% "
        summary += f"({'Inflasi' if avg_iph > 0 else 'Deflasi'})\n"
        
        if commodity_stats:
            top_3 = commodity_stats[:3]
            summary += f"• Top 3 komoditas berpengaruh:\n"
            for i, comm in enumerate(top_3, 1):
                summary += f"  {i}. {comm['name']} (dampak total: {comm['total_impact']:.3f}%)\n"
        
        # Identify pattern
        if abs(avg_iph) > 2:
            summary += f"⚠️ Perhatian: {month} menunjukkan pergerakan harga signifikan!"
        
        return summary
    
    def get_commodity_trends(self, commodity_name=None, periods=4):
        """Analyze trends for specific commodity or all commodities"""
        df = self.load_commodity_data()
        
        if df.empty:
            return {'success': False, 'message': 'No data available'}
        
        # Get last N periods
        df_recent = df.tail(periods)
        
        trends = {}
        
        for _, row in df_recent.iterrows():
            period = f"{row['Bulan']} {row['Minggu']}"
            commodities = self.parse_commodity_impacts(row.get('Komoditas_Andil', ''))
            
            for comm in commodities:
                if commodity_name and comm['name'] != commodity_name:
                    continue
                    
                if comm['name'] not in trends:
                    trends[comm['name']] = {
                        'periods': [],
                        'impacts': [],
                        'appearances': 0
                    }
                
                trends[comm['name']]['periods'].append(period)
                trends[comm['name']]['impacts'].append(comm['impact'])
                trends[comm['name']]['appearances'] += 1
        
        # Calculate trend direction
        for name, data in trends.items():
            if len(data['impacts']) > 1:
                # Simple linear trend
                x = np.arange(len(data['impacts']))
                z = np.polyfit(x, data['impacts'], 1)
                data['trend_coefficient'] = float(z[0])
                
                if abs(z[0]) < 0.01:
                    data['trend'] = 'stable'
                elif z[0] > 0:
                    data['trend'] = 'increasing'
                else:
                    data['trend'] = 'decreasing'
            else:
                data['trend'] = 'insufficient_data'
                data['trend_coefficient'] = 0
        
        return {
            'success': True,
            'periods_analyzed': periods,
            'commodity_trends': trends,
            'summary': self._generate_trend_summary(trends)
        }
    
    def _generate_trend_summary(self, trends):
        """Generate trend summary"""
        if not trends:
            return "No commodity trends available"
        
        increasing = [name for name, data in trends.items() if data.get('trend') == 'increasing']
        decreasing = [name for name, data in trends.items() if data.get('trend') == 'decreasing']
        
        summary = "📈 Trend Komoditas:\n"
        
        if increasing:
            summary += f"• Dampak meningkat: {', '.join(increasing[:3])}\n"
        
        if decreasing:
            summary += f"• Dampak menurun: {', '.join(decreasing[:3])}\n"
        
        # Most consistent commodity
        most_consistent = max(trends.items(), key=lambda x: x[1]['appearances'])
        summary += f"• Paling konsisten muncul: {most_consistent[0]} ({most_consistent[1]['appearances']} kali)"
        
        return summary
    
    def get_alert_commodities(self, threshold=0.05):
        """Get commodities that need attention based on volatility"""
        df = self.load_commodity_data()
        
        if df.empty:
            return {'success': False, 'message': 'No data available'}
        
        # Get recent high volatility commodities
        recent_df = df.tail(4)  # Last 4 weeks
        
        alerts = []
        
        for _, row in recent_df.iterrows():
            if row.get('Nilai_Fluktuasi', 0) > threshold:
                alerts.append({
                    'period': f"{row['Bulan']} {row['Minggu']}",
                    'commodity': row.get('Komoditas_Fluktuasi_Tertinggi', 'Unknown'),
                    'volatility': float(row.get('Nilai_Fluktuasi', 0)),
                    'iph_impact': float(row.get('IPH', 0)),
                    'severity': 'high' if row.get('Nilai_Fluktuasi', 0) > 0.08 else 'medium'
                })
        
        # Sort by volatility
        alerts.sort(key=lambda x: x['volatility'], reverse=True)
        
        return {
            'success': True,
            'threshold': threshold,
            'alerts': alerts,
            'summary': f"Found {len(alerts)} commodities with volatility > {threshold:.1%}"
        }
    
    def get_seasonal_patterns(self):
        """Identify seasonal patterns in commodity impacts"""
        df = self.load_commodity_data()
        
        if df.empty:
            return {'success': False, 'message': 'No data available'}
        
        # Group by month
        monthly_patterns = {}
        
        for month in df['Bulan'].unique():
            month_data = df[df['Bulan'] == month]
            
            # Collect all commodities for this month
            month_commodities = defaultdict(list)
            
            for _, row in month_data.iterrows():
                commodities = self.parse_commodity_impacts(row.get('Komoditas_Andil', ''))
                for comm in commodities:
                    month_commodities[comm['name']].append(abs(comm['impact']))
            
            # Find dominant commodities
            dominant = []
            for name, impacts in month_commodities.items():
                dominant.append({
                    'name': name,
                    'avg_impact': np.mean(impacts),
                    'frequency': len(impacts)
                })
            
            dominant.sort(key=lambda x: x['avg_impact'], reverse=True)
            
            monthly_patterns[month] = {
                'avg_iph': float(month_data['IPH'].mean()),
                'dominant_commodities': dominant[:3],
                'volatility': float(month_data['IPH'].std()) if len(month_data) > 1 else 0
            }
        
        return {
            'success': True,
            'seasonal_patterns': monthly_patterns,
            'summary': self._generate_seasonal_summary(monthly_patterns)
        }
    
    def _generate_seasonal_summary(self, patterns):
        """Generate seasonal pattern summary"""
        summary = "🗓️ Pola Seasonal Komoditas:\n"
        
        # Find months with highest inflation/deflation
        sorted_months = sorted(patterns.items(), key=lambda x: x[1]['avg_iph'], reverse=True)
        
        if sorted_months:
            highest = sorted_months[0]
            lowest = sorted_months[-1]
            
            summary += f"• Inflasi tertinggi: {highest[0]} ({highest[1]['avg_iph']:.2f}%)\n"
            summary += f"• Deflasi tertinggi: {lowest[0]} ({lowest[1]['avg_iph']:.2f}%)\n"
        
        # Find most volatile month
        volatile_month = max(patterns.items(), key=lambda x: x[1]['volatility'])
        summary += f"• Bulan paling volatile: {volatile_month[0]} (std: {volatile_month[1]['volatility']:.3f}%)"
        
        return summary