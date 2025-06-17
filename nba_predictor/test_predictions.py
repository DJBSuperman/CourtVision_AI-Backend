# test_predictions.py
import requests
import json
import pandas as pd

BASE_URL = 'http://localhost:8000/api/'

def test_prediction(home_team, away_team):
    url = f"{BASE_URL}predict-custom/"
    data = {
        'home_team': home_team,
        'away_team': away_team
    }
    
    print(f"\n===== TESTING: {home_team} vs {away_team} =====")
    print(f"Sending request to: {url}")
    print(f"Request data: {data}")
    
    try:
        response = requests.post(url, json=data)
        
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Display key information
            home_win_prob = result['prediction']['home_win_probability']
            away_win_prob = 1 - home_win_prob
            
            print(f"\nPREDICTION RESULTS:")
            print(f"Home team ({home_team}) win probability: {home_win_prob:.4f} ({home_win_prob*100:.1f}%)")
            print(f"Away team ({away_team}) win probability: {away_win_prob:.4f} ({away_win_prob*100:.1f}%)")
            
            if 'raw_probability' in result['prediction']:
                print(f"Raw probability (before calibration): {result['prediction']['raw_probability']}")
            
            print(f"\nPredicted winner: {result['prediction'].get('predicted_winner', 'Not specified')}")
            
            # Display team advantages
            print("\nTEAM ADVANTAGES:")
            if 'top_factors' in result['prediction']:
                print("Top factors from prediction:")
                for factor in result['prediction']['top_factors']:
                    print(f"  - {factor['feature']}: {factor['team_favored']} team advantage")
            
            if 'home_team_advantages' in result.get('analysis', {}):
                print(f"\nHome team advantages: {result['analysis']['home_team_advantages']}")
                
            if 'away_team_advantages' in result.get('analysis', {}):
                print(f"Away team advantages: {result['analysis']['away_team_advantages']}")
            
            # Save full response to file for inspection
            filename = f"prediction_{home_team}_vs_{away_team}.json"
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nFull response saved to {filename}")
            
            return result
        else:
            print("Error:", response.text)
            return None
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return None

# Test with different matchups
results = []
teams_to_test = [
    ('LAL', 'BOS'),  # Lakers vs Celtics - historically competitive
    ('MIA', 'GSW'),  # Heat vs Warriors - different conferences
    ('CHI', 'PHI'),  # Bulls vs 76ers - similar strength
    ('NYK', 'BKN'),  # Knicks vs Nets - local rivalry
    ('MIL', 'DET')   # Bucks vs Pistons - strong vs weaker team
]

for home, away in teams_to_test:
    result = test_prediction(home, away)
    if result:
        results.append({
            'matchup': f"{home} vs {away}",
            'home_win_probability': result['prediction']['home_win_probability'],
            'predicted_winner': result['prediction'].get('predicted_winner', 'Unknown')
        })

# Create a comparison table
if results:
    print("\n===== COMPARISON OF PREDICTIONS =====")
    df = pd.DataFrame(results)
    print(df)