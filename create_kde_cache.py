import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from tkinter import filedialog
import os

year = "2025"

def processing_full(dataframe):
    # Filter called pitches
    df = dataframe[dataframe['PitchCall'].isin(['BallCalled', 'StrikeCalled'])].copy()

    if df.empty:
        return df  # Return early if there's nothing to process

    # Convert to inches and flip side for catcher's POV
    df['PlateLocHeightIn'] = 12 * df['PlateLocHeight']
    df['PlateLocSideIn'] = -12 * df['PlateLocSide']

    # Map calls to numeric values
    call_map = {'BallCalled': 0, 'StrikeCalled': 1}
    df['CallValue'] = df['PitchCall'].map(call_map)

    # Drop NAs for relevant columns
    df = df[['PitchCall', 'PlateLocHeightIn', 'PlateLocSideIn', 'Catcher', 'BatterSide', 'CallValue']].dropna()

    return df

def create_kde_cache(league_df, cache_file=f"kde_cache_{year}.npz"):
    """
    Pre-compute and save KDE probability grids for league data
    """
    print("Computing KDE cache for league data...")
    print(f"Total pitches in master data: {len(league_df)}")
    
    # Process league data
    league_proc_df = processing_full(league_df)
    print(f"Processed called pitches: {len(league_proc_df)}")
    
    if league_proc_df.empty:
        print("No valid called pitches found in master data!")
        return False
    
    # Create grid for heatmap
    x_grid = np.linspace(-20, 20, 100)
    y_grid = np.linspace(10, 50, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])
    
    # Split by batter side
    league_l = league_proc_df[league_proc_df['BatterSide'] == 'Left']
    league_r = league_proc_df[league_proc_df['BatterSide'] == 'Right']
    
    print(f"Left-handed batter pitches: {len(league_l)}")
    print(f"Right-handed batter pitches: {len(league_r)}")
    
    # Calculate KDE for each side
    def calculate_strike_probability(strikes, balls, positions):
        if len(strikes) == 0 and len(balls) == 0:
            return np.zeros(positions.shape[1])
        
        total_pitches = len(strikes) + len(balls)
        if total_pitches == 0:
            return np.zeros(positions.shape[1])
        
        # KDE for strikes
        if len(strikes) > 0:
            kde_strikes = gaussian_kde(strikes, bw_method='silverman')
            strike_density = kde_strikes(positions)
        else:
            strike_density = np.zeros(positions.shape[1])
        
        # KDE for balls
        if len(balls) > 0:
            kde_balls = gaussian_kde(balls, bw_method='silverman')
            ball_density = kde_balls(positions)
        else:
            ball_density = np.zeros(positions.shape[1])
        
        # Calculate strike probability
        total_density = strike_density + ball_density
        strike_prob = np.where(total_density > 0, strike_density / total_density, 0)
        
        return strike_prob
    
    print("Calculating KDE for left-handed batters...")
    # Calculate for left-handed batters
    league_strikes_l = league_l[league_l['CallValue'] == 1][['PlateLocSideIn', 'PlateLocHeightIn']].values.T
    league_balls_l = league_l[league_l['CallValue'] == 0][['PlateLocSideIn', 'PlateLocHeightIn']].values.T
    league_strike_prob_l = calculate_strike_probability(league_strikes_l, league_balls_l, positions)
    
    print("Calculating KDE for right-handed batters...")
    # Calculate for right-handed batters
    league_strikes_r = league_r[league_r['CallValue'] == 1][['PlateLocSideIn', 'PlateLocHeightIn']].values.T
    league_balls_r = league_r[league_r['CallValue'] == 0][['PlateLocSideIn', 'PlateLocHeightIn']].values.T
    league_strike_prob_r = calculate_strike_probability(league_strikes_r, league_balls_r, positions)
    
    # Save to file
    np.savez(cache_file,
             league_strike_prob_l=league_strike_prob_l,
             league_strike_prob_r=league_strike_prob_r,
             x_grid=x_grid,
             y_grid=y_grid)
    
    print(f"KDE cache saved to {cache_file}")
    print(f"Cache file size: {os.path.getsize(cache_file) / 1024 / 1024:.2f} MB")
    
    # Print some statistics
    print("\nCache Statistics:")
    print(f"Left-handed strike rate: {np.mean(league_strike_prob_l):.3f}")
    print(f"Right-handed strike rate: {np.mean(league_strike_prob_r):.3f}")
    print(f"Grid resolution: {len(x_grid)} x {len(y_grid)} points")
    
    return True

def main():
    print("KDE Cache Creation Script")
    print("=" * 40)
    
    # Ask user to select master CSV file
    print("Please select your master CSV file containing all pitch data...")
    master_csv = filedialog.askopenfilename(
        title="Select Master CSV file", 
        filetypes=(("CSV files", "*.csv"), ("all files", "*.*"))
    )
    
    if not master_csv:
        print("No file selected. Exiting.")
        return
    
    print(f"Loading master CSV: {master_csv}")
    
    try:
        # Load the master CSV
        league_df = pd.read_csv(master_csv)
        print(f"Successfully loaded {len(league_df)} rows from master CSV")
        
        # Check required columns
        required_columns = ['PlateLocSide', 'PlateLocHeight', 'PitchCall', 'BatterSide']
        missing_columns = [col for col in required_columns if col not in league_df.columns]
        
        if missing_columns:
            print(f"Warning: Missing required columns: {missing_columns}")
            print("Available columns:", list(league_df.columns))
            return
        
        # Create cache
        success = create_kde_cache(league_df)
        
        if success:
            print("\n✅ Cache creation completed successfully!")
            print("You can now run the main catcher report script.")
        else:
            print("\n❌ Cache creation failed!")
            
    except Exception as e:
        print(f"Error loading or processing CSV: {e}")
        return

if __name__ == "__main__":
    main() 