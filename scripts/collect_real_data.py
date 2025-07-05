#!/usr/bin/env python3
"""
Collect real Premier League data from football-data.co.uk.

This script collects historical Premier League match data from football-data.co.uk
for the past 8 seasons and saves it in a standardized format.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_collection.data_collector import AlternativeDataCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to collect Premier League data."""
    print("🏈 Premier League Data Collection")
    print("=" * 40)
    print("Collecting data from football-data.co.uk for the past 8 seasons...")
    print()
    
    # Create collector
    collector = AlternativeDataCollector()
    
    # Collect data
    df = collector.collect_all_data()
    
    if df is not None and len(df) > 0:
        # Save data
        output_dir = Path("data/real_data")
        collector.save_data(df, output_dir)
        
        print()
        print("✅ Data Collection Complete!")
        print(f"📊 Successfully collected {len(df)} matches")
        print(f"💾 Data saved to {output_dir}")
        
        # Show statistics
        print("\n📈 Data Statistics:")
        print(f"  • Total matches: {len(df)}")
        if 'date' in df.columns:
            print(f"  • Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        if 'season' in df.columns:
            print(f"  • Seasons: {', '.join(sorted(df['season'].unique()))}")
        if 'home_team' in df.columns and 'away_team' in df.columns:
            print(f"  • Teams: {len(set(df['home_team'].unique()) | set(df['away_team'].unique()))}")
        
        print(f"  • Columns: {', '.join(df.columns)}")
        
        # Show sample data
        print("\n📋 Sample Data:")
        print(df.head(3).to_string())
        
        return df
    else:
        print("❌ Failed to collect data")
        return None


if __name__ == "__main__":
    main()
