import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataHandler:
    """Handler for data operations including loading, validation, and storage"""
    
    def __init__(self, data_path='data/historical_data.csv', backup_path='data/backups/'):
        self.data_path = data_path
        self.backup_path = backup_path
        
        # Create directories
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        os.makedirs(self.backup_path, exist_ok=True)
        
        print(f"📁 DataHandler initialized:")
        print(f"   📊 Data path: {self.data_path}")
        print(f"   💾 Backup path: {self.backup_path}")
    
    def load_historical_data(self):
        """Load historical data from CSV file"""
        if os.path.exists(self.data_path):
            try:
                df = pd.read_csv(self.data_path)
                
                # Ensure proper data types
                if 'Tanggal' in df.columns:
                    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
                
                if 'Indikator_Harga' in df.columns:
                    df['Indikator_Harga'] = pd.to_numeric(df['Indikator_Harga'], errors='coerce')
                
                # Sort by date
                df = df.sort_values('Tanggal').reset_index(drop=True)
                
                print(f"✅ Loaded {len(df)} historical records")
                print(f"   📅 Date range: {df['Tanggal'].min().strftime('%Y-%m-%d')} to {df['Tanggal'].max().strftime('%Y-%m-%d')}")
                
                return df
                
            except Exception as e:
                print(f"❌ Error loading historical data: {str(e)}")
                return pd.DataFrame()
        else:
            print("ℹ️ No historical data file found")
            return pd.DataFrame()
    
    def validate_new_data(self, df):
        """Validate new data format and content"""
        print("🔍 Validating new data...")
        
        if df.empty:
            raise ValueError("❌ Data is empty")
        
        print(f"📊 Original shape: {df.shape}")
        print(f"📋 Original columns: {list(df.columns)}")
        
        # Check for completely empty rows
        empty_rows = df.isnull().all(axis=1).sum()
        if empty_rows > 0:
            print(f"🧹 Removing {empty_rows} completely empty rows")
            df = df.dropna(how='all')
        
        # Map common column variations to standard names
        column_mapping = {
            'Indikator Perubahan Harga (%)': 'Indikator_Harga',
            ' Indikator Perubahan Harga (%)': 'Indikator_Harga',
            'Indikator_Perubahan_Harga': 'Indikator_Harga',
            'IPH': 'Indikator_Harga',
            'Date': 'Tanggal',
            'date': 'Tanggal',
            'Tanggal ': 'Tanggal',  # Handle trailing space
            ' Tanggal': 'Tanggal'   # Handle leading space
        }
        
        # Rename columns based on mapping
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
                print(f"✅ Renamed column '{old_name}' to '{new_name}'")
        
        # Special handling for IPH Kota Batu format
        if 'Tanggal' not in df.columns and 'Bulan' in df.columns and 'Minggu ke-' in df.columns:
            print("🔄 Creating Tanggal column from Bulan and Minggu...")
            df = self._create_date_from_bulan_minggu(df)
        
        # Check required columns
        required_columns = ['Tanggal', 'Indikator_Harga']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            available_cols = list(df.columns)
            raise ValueError(f"❌ Missing required columns: {missing_cols}. Available columns: {available_cols}")
        
        print(f"✅ Required columns found: {required_columns}")
        
        # Validate and convert date column
        try:
            original_dates = len(df)
            df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
            invalid_dates = df['Tanggal'].isna().sum()
            
            if invalid_dates > 0:
                print(f"⚠️ Found {invalid_dates}/{original_dates} invalid dates, removing...")
                df = df.dropna(subset=['Tanggal'])
                
            if df.empty:
                raise ValueError("❌ No valid dates found in data")
                
            print(f"📅 Date validation: {len(df)}/{original_dates} valid dates")
            
        except Exception as e:
            raise ValueError(f"❌ Error processing dates: {str(e)}")
        
        # Validate and convert numeric column
        try:
            original_values = len(df)
            df['Indikator_Harga'] = pd.to_numeric(df['Indikator_Harga'], errors='coerce')
            invalid_values = df['Indikator_Harga'].isna().sum()
            
            if invalid_values > 0:
                print(f"⚠️ Found {invalid_values}/{original_values} invalid numeric values, removing...")
                df = df.dropna(subset=['Indikator_Harga'])
                
            if df.empty:
                raise ValueError("❌ No valid numeric values found")
                
            print(f"📊 Numeric validation: {len(df)}/{original_values} valid values")
            
        except Exception as e:
            raise ValueError(f"❌ Error processing numeric values: {str(e)}")
        
        # Check for reasonable value ranges
        iph_values = df['Indikator_Harga']
        extreme_values = ((iph_values < -50) | (iph_values > 50)).sum()
        
        if extreme_values > 0:
            print(f"⚠️ Warning: {extreme_values} extreme IPH values (>50% or <-50%)")
            
        # Sort by date
        df = df.sort_values('Tanggal').reset_index(drop=True)
        
        print(f"✅ Validation complete: {len(df)} valid records")
        print(f"   📊 IPH range: {iph_values.min():.2f}% to {iph_values.max():.2f}%")
        print(f"   📅 Date range: {df['Tanggal'].min().strftime('%Y-%m-%d')} to {df['Tanggal'].max().strftime('%Y-%m-%d')}")
        
        return df

    def _create_date_from_bulan_minggu(self, df):
        """Create Tanggal column from Bulan and Minggu ke- columns"""
        
        def extract_year_from_bulan(bulan_str):
            """Extract year from month string"""
            if pd.isna(bulan_str):
                return 2024  # Default year
            
            bulan_str = str(bulan_str).strip()
            if "'24" in bulan_str:
                return 2024
            elif "'25" in bulan_str:
                return 2025
            elif "'23" in bulan_str:
                return 2023
            else:
                return 2024  # Default year
        
        def extract_month_from_bulan(bulan_str):
            """Extract month number from month name"""
            month_map = {
                'januari': 1, 'februari': 2, 'maret': 3, 'april': 4,
                'mei': 5, 'juni': 6, 'juli': 7, 'agustus': 8,
                'september': 9, 'oktober': 10, 'november': 11, 'desember': 12
            }
            
            if pd.isna(bulan_str):
                return 1
            
            bulan_str = str(bulan_str).strip().lower()
            bulan_clean = bulan_str.split("'")[0].strip()
            
            for nama_bulan, nomor_bulan in month_map.items():
                if nama_bulan in bulan_clean:
                    return nomor_bulan
            return 1
        
        def extract_week_from_minggu(minggu_str):
            """Extract week number from week string"""
            if pd.isna(minggu_str):
                return 1
            
            minggu_str = str(minggu_str).strip().upper()
            if minggu_str.startswith('M'):
                week_num = minggu_str.replace('M', '')
            else:
                week_num = minggu_str
            
            try:
                return int(week_num)
            except:
                return 1
        
        def create_date(year, month, week):
            """Create date from year, month, and week"""
            try:
                # Calculate day based on week (week 1 = day 1-7, week 2 = day 8-14, etc.)
                day = (week - 1) * 7 + 1
                
                # Ensure day doesn't exceed month limits
                if month == 2:  # February
                    max_day = 29 if year % 4 == 0 else 28
                elif month in [4, 6, 9, 11]:  # April, June, September, November
                    max_day = 30
                else:
                    max_day = 31
                
                if day > max_day:
                    day = max_day
                
                return pd.Timestamp(year, month, day)
            except:
                return pd.Timestamp(year, month, 1)
        
        # Extract components
        df['Year'] = df['Bulan'].apply(extract_year_from_bulan)
        df['Month'] = df['Bulan'].apply(extract_month_from_bulan)
        df['Week'] = df['Minggu ke-'].apply(extract_week_from_minggu)
        
        # Create Tanggal column
        df['Tanggal'] = df.apply(lambda row: create_date(row['Year'], row['Month'], row['Week']), axis=1)
        
        # Clean up temporary columns
        df = df.drop(['Year', 'Month', 'Week'], axis=1)
        
        print(f"✅ Created Tanggal column from Bulan and Minggu")
        
        return df

    def backup_current_data(self):
        """Create backup of current data before modifications"""
        if os.path.exists(self.data_path):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"historical_data_backup_{timestamp}.csv"
            backup_filepath = os.path.join(self.backup_path, backup_filename)
            
            try:
                shutil.copy2(self.data_path, backup_filepath)
                
                # Also create a quick info file
                info_filepath = backup_filepath.replace('.csv', '_info.txt')
                with open(info_filepath, 'w') as f:
                    f.write(f"Backup created: {datetime.now().isoformat()}\n")
                    f.write(f"Original file: {self.data_path}\n")
                    f.write(f"File size: {os.path.getsize(self.data_path)} bytes\n")
                
                print(f"✅ Data backed up to {backup_filename}")
                return backup_filepath
                
            except Exception as e:
                print(f"❌ Error backing up data: {str(e)}")
                return None
        return None
    
    def merge_and_save_data(self, new_data_df):
        """Merge new data with historical data and save"""
        print("🔄 Starting data merge process...")
        
        # Validate new data
        validated_df = self.validate_new_data(new_data_df.copy())
        
        # Create backup of current data
        backup_path = self.backup_current_data()
        
        # Load existing data
        existing_df = self.load_historical_data()
        
        if existing_df.empty:
            # No existing data, use new data as base
            combined_df = validated_df.copy()
            print("ℹ️ No existing data. Using new data as historical data.")
            merge_info = {
                'existing_records': 0,
                'new_records': len(validated_df),
                'total_records': len(combined_df),
                'duplicates_removed': 0,
                'date_overlap': False
            }
        else:
            # Check for overlapping dates
            existing_dates = set(existing_df['Tanggal'].dt.date)
            new_dates = set(validated_df['Tanggal'].dt.date)
            overlap_dates = existing_dates.intersection(new_dates)
            
            if overlap_dates:
                print(f"⚠️ Found {len(overlap_dates)} overlapping dates")
                print(f"   📅 Overlap range: {min(overlap_dates)} to {max(overlap_dates)}")
            
            # Merge data
            combined_df = pd.concat([existing_df, validated_df], ignore_index=True)
            
            # Remove duplicates (keep latest values for same dates)
            initial_count = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=['Tanggal'], keep='last')
            duplicates_removed = initial_count - len(combined_df)
            
            # Sort by date
            combined_df = combined_df.sort_values('Tanggal').reset_index(drop=True)
            
            merge_info = {
                'existing_records': len(existing_df),
                'new_records': len(validated_df),
                'total_records': len(combined_df),
                'duplicates_removed': duplicates_removed,
                'date_overlap': len(overlap_dates) > 0,
                'overlap_count': len(overlap_dates)
            }
            
            print(f"✅ Data merged successfully:")
            print(f"   📊 {merge_info['existing_records']} existing + {merge_info['new_records']} new = {merge_info['total_records']} total")
            if duplicates_removed > 0:
                print(f"   🔄 Removed {duplicates_removed} duplicate records")
        
        # Add metadata columns if not present
        self._add_metadata_columns(combined_df)
        
        # Validate final dataset
        self._validate_final_dataset(combined_df)
        
        # Save combined data
        try:
            combined_df.to_csv(self.data_path, index=False)
            print(f"✅ Combined data saved to {self.data_path}")
            
            # Save merge info
            self._save_merge_info(merge_info, backup_path)
            
            return combined_df, merge_info
            
        except Exception as e:
            print(f"❌ Error saving combined data: {str(e)}")
            # Attempt to restore backup
            if backup_path and os.path.exists(backup_path):
                try:
                    shutil.copy2(backup_path, self.data_path)
                    print("🔄 Restored data from backup")
                except:
                    pass
            raise
    
    def _add_metadata_columns(self, df):
        """Add metadata columns to dataset"""
        # Add periodo column if not present
        if 'Periode' not in df.columns:
            df['Periode'] = range(1, len(df) + 1)
        
        # Add additional time-based columns
        if 'Tanggal' in df.columns:
            df['Year'] = df['Tanggal'].dt.year
            df['Month'] = df['Tanggal'].dt.month
            df['Quarter'] = df['Tanggal'].dt.quarter
            df['WeekOfYear'] = df['Tanggal'].dt.isocalendar().week
        
        # Add data quality indicators
        df['Data_Source'] = 'uploaded'
        df['Last_Updated'] = datetime.now().isoformat()
    
    def _validate_final_dataset(self, df):
        """Final validation of the complete dataset"""
        print("🔍 Performing final dataset validation...")
        
        # Check for gaps in time series
        if len(df) > 1:
            df_sorted = df.sort_values('Tanggal')
            date_diffs = df_sorted['Tanggal'].diff().dt.days
            
            # Look for unusual gaps (more than 14 days)
            large_gaps = date_diffs[date_diffs > 14]
            if not large_gaps.empty:
                print(f"⚠️ Found {len(large_gaps)} time gaps > 14 days")
        
        # Check for outliers
        iph_values = df['Indikator_Harga']
        Q1 = iph_values.quantile(0.25)
        Q3 = iph_values.quantile(0.75)
        IQR = Q3 - Q1
        outliers = iph_values[(iph_values < Q1 - 1.5*IQR) | (iph_values > Q3 + 1.5*IQR)]
        
        if not outliers.empty:
            print(f"⚠️ Detected {len(outliers)} potential outliers")
            print(f"   📊 Outlier range: {outliers.min():.2f}% to {outliers.max():.2f}%")
        
        # Check data completeness
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        print(f"✅ Dataset completeness: {completeness:.1f}%")
        
        # Statistical summary
        print(f"📊 Final dataset statistics:")
        print(f"   📈 Mean IPH: {iph_values.mean():.3f}%")
        print(f"   📉 Std IPH: {iph_values.std():.3f}%")
        print(f"   📊 Range: {iph_values.min():.2f}% to {iph_values.max():.2f}%")
    
    def _save_merge_info(self, merge_info, backup_path):
        """Save information about the merge operation"""
        try:
            info_file = os.path.join(self.backup_path, f"merge_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            merge_info['timestamp'] = datetime.now().isoformat()
            merge_info['backup_path'] = backup_path
            merge_info['final_data_path'] = self.data_path
            
            import json
            with open(info_file, 'w') as f:
                json.dump(merge_info, f, indent=2, default=str)
                
        except Exception as e:
            print(f"⚠️ Could not save merge info: {str(e)}")
    
    def get_data_summary(self):
        """Get comprehensive summary of current data"""
        df = self.load_historical_data()
        
        if df.empty:
            return {
                'total_records': 0,
                'date_range': None,
                'latest_value': None,
                'statistics': {},
                'data_quality': {},
                'file_info': {}
            }
        
        # Basic statistics
        iph_values = df['Indikator_Harga']
        
        # Data quality metrics
        missing_values = df.isnull().sum().sum()
        completeness = (1 - missing_values / (len(df) * len(df.columns))) * 100
        
        # Detect outliers
        Q1 = iph_values.quantile(0.25)
        Q3 = iph_values.quantile(0.75)
        IQR = Q3 - Q1
        outliers = iph_values[(iph_values < Q1 - 1.5*IQR) | (iph_values > Q3 + 1.5*IQR)]
        
        # File information
        file_info = {}
        if os.path.exists(self.data_path):
            stat = os.stat(self.data_path)
            file_info = {
                'size_mb': stat.st_size / (1024 * 1024),
                'last_modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
            }
        
        summary = {
            'total_records': int(len(df)),  # Convert to int
            'date_range': {
                'start': df['Tanggal'].min().strftime('%Y-%m-%d'),
                'end': df['Tanggal'].max().strftime('%Y-%m-%d'),
                'days_span': int((df['Tanggal'].max() - df['Tanggal'].min()).days)  # Convert to int
            },
            'latest_value': float(iph_values.iloc[-1]),  # Convert to float
            'statistics': {
                'mean': float(iph_values.mean()),
                'std': float(iph_values.std()),
                'min': float(iph_values.min()),
                'max': float(iph_values.max()),
                'median': float(iph_values.median()),
                'q1': float(Q1),
                'q3': float(Q3)
            },
            'data_quality': {
                'completeness_percent': float(completeness),
                'missing_values': int(missing_values),
                'outliers_count': int(len(outliers)),
                'outliers_percent': float(len(outliers) / len(df) * 100)
            },
            'file_info': file_info,
            'columns': list(df.columns),
            'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        
        return summary
    
    def get_recent_backups(self, limit=10):
        """Get list of recent backup files"""
        if not os.path.exists(self.backup_path):
            return []
        
        backup_files = []
        for file in os.listdir(self.backup_path):
            if file.startswith('historical_data_backup_') and file.endswith('.csv'):
                filepath = os.path.join(self.backup_path, file)
                stat = os.stat(filepath)
                
                backup_files.append({
                    'filename': file,
                    'filepath': filepath,
                    'size_mb': stat.st_size / (1024 * 1024),
                    'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        # Sort by creation time (newest first)
        backup_files.sort(key=lambda x: x['created'], reverse=True)
        
        return backup_files[:limit]
    
    def restore_from_backup(self, backup_filename):
        """Restore data from a specific backup"""
        backup_filepath = os.path.join(self.backup_path, backup_filename)
        
        if not os.path.exists(backup_filepath):
            raise FileNotFoundError(f"Backup file not found: {backup_filename}")
        
        try:
            # Create backup of current data first
            current_backup = self.backup_current_data()
            
            # Restore from backup
            shutil.copy2(backup_filepath, self.data_path)
            
            print(f"✅ Data restored from backup: {backup_filename}")
            print(f"💾 Current data backed up to: {os.path.basename(current_backup)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error restoring from backup: {str(e)}")
            return False