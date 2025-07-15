
import sys
from pathlib import Path
from prefect import flow, serve
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

logger = logging.getLogger(__name__)

@flow(name="data_pipeline_flow", log_prints=True)
def data_pipeline_flow(years_back: int = 3):
    """Real data pipeline flow - fetches actual Premier League data"""
    print(f"🔄 Starting real data pipeline (fetching {years_back} years of data)")
    
    try:
        # Import the real data fetcher
        from src.data_integration.football_data_fetcher import FootballDataFetcher
        
        # Initialize fetcher
        fetcher = FootballDataFetcher()
        print(f"📡 Fetching Premier League data for last {years_back} years...")
        
        # Get historical data
        df = fetcher.get_historical_data(years_back=years_back)
        
        if df.empty:
            print("❌ No data fetched")
            return {"status": "failed", "error": "No data retrieved"}
        
        # Save processed data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = fetcher.save_processed_data(df, f"premier_league_data_{timestamp}.csv")
        
        print(f"✅ Successfully fetched {len(df)} matches")
        print(f"📁 Data saved to: {filepath}")
        
        result = {
            "status": "completed",
            "matches_fetched": len(df),
            "file_path": str(filepath),
            "seasons": df['season'].unique().tolist() if 'season' in df.columns else [],
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"🎉 Data pipeline completed: {result}")
        return result
        
    except Exception as e:
        print(f"❌ Data pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    serve(
        data_pipeline_flow.to_deployment(
            name="data_pipeline_flow-deployment",
            work_pool_name="premier-league-pool"
        )
    )
