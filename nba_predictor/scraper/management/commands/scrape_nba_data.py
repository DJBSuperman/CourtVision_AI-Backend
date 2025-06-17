# scraper/management/commands/scrape_nba_data.py
import os
import time
import pandas as pd
import logging
from django.core.management.base import BaseCommand
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from io import StringIO

from predictor_api.models import Team, TeamStats, Game

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Scrape NBA data from basketball-reference.com'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--schedule',
            action='store_true',
            help='Scrape NBA schedule data'
        )
        parser.add_argument(
            '--standings',
            action='store_true',
            help='Scrape NBA standings data'
        )
        parser.add_argument(
            '--stats',
            action='store_true',
            help='Scrape NBA team statistics data'
        )
        parser.add_argument(
            '--all',
            action='store_true',
            help='Scrape all NBA data'
        )
        parser.add_argument(
            '--save-csv',
            action='store_true',
            help='Save scraped data as CSV files'
        )
    
    def handle(self, *args, **options):
        if options['all'] or options['schedule']:
            self.scrape_schedule(save_csv=options['save_csv'])
        if options['all'] or options['standings']:
            self.scrape_standings(save_csv=options['save_csv'])
        if options['all'] or options['stats']:
            self.scrape_team_stats(save_csv=options['save_csv'])
    
    def setup_driver(self):
        """Set up Chrome WebDriver"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        service = Service()  # Will use default ChromeDriver path
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        return driver
    
    def scrape_schedule(self, save_csv=False):
        """Scrape NBA schedule data for the current season"""
        self.stdout.write("Scraping NBA schedule data...")
        
        # Base URL for NBA schedule
        months = ["october", "november", "december", "january", "february", "march", "april", "may"]
        base_url = "https://www.basketball-reference.com/leagues/NBA_2025_games-{}.html"
        
        dfs = []
        driver = self.setup_driver()
        
        for month in months:
            try:
                url = base_url.format(month)
                self.stdout.write(f"Scraping schedule for {month}...")
                
                driver.get(url)
                time.sleep(3)  # Wait for the page to load
                
                # Get the page source and parse with BeautifulSoup
                soup = BeautifulSoup(driver.page_source, "html.parser")
                
                # Remove elements that might interfere with table parsing
                section_heading = soup.find('div', id="schedule_sh")
                if section_heading:
                    section_heading.decompose()
                
                # Find the schedule table
                schedule_table = soup.find(id="all_schedule")
                
                if schedule_table:
                    # Use pandas to read the HTML table
                    month_df = pd.read_html(str(schedule_table))[0]
                    dfs.append(month_df)
                    self.stdout.write(f"Successfully scraped {len(month_df)} games for {month}")
                else:
                    self.stdout.write(self.style.WARNING(f"No schedule table found for {month}"))
            
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error scraping {month} schedule: {str(e)}"))
                logger.error(f"Error scraping {month} schedule: {str(e)}")
        
        driver.quit()
        
        if not dfs:
            self.stdout.write(self.style.ERROR("No schedule data scraped"))
            return
        
        # Combine all months' data
        schedule_df = pd.concat(dfs, ignore_index=True)
        
         # Clean and process the data
        schedule_df.columns = ["Date", "Start (ET)", "Visitor/Neutral", "PTS", "Home/Neutral", "PTS.1", "Unnamed: 6", "Unnamed: 7", "Attend", "LOG", "Arena", "Notes"]
        
        # Select and rename required columns
        df_cleaned = schedule_df[[
            "Date", "Visitor/Neutral", "PTS", "Home/Neutral", "PTS.1", 
            "Attend", "Arena"
        ]]
        
        # Add game ID as the first column
        df_cleaned.insert(0, "game_id", df_cleaned.index + 1)
        
        # Fill missing values in score-related columns with 0
        df_cleaned[["PTS", "PTS.1", "Attend"]] = df_cleaned[["PTS", "PTS.1", "Attend"]].fillna(0)
        
        # Save as CSV if requested
        if save_csv:
            csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'cleaned_nba_schedule.csv')
            df_cleaned.to_csv(csv_path, index=False)
            self.stdout.write(f"Saved schedule data to {csv_path}")
        
        # Update team mapping for database
        team_name_mapping = {
            "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BRK", "Charlotte Hornets": "CHO",
            "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
            "Detroit Pistons": "DET", "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
            "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM", "Miami Heat": "MIA",
            "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
            "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHO",
            "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
            "Utah Jazz": "UTA", "Washington Wizards": "WAS"
        }
        
        # Ensure all teams exist in the database
        for team_name, team_code in team_name_mapping.items():
            Team.objects.get_or_create(code=team_code, defaults={'name': team_name})
        
        # Import games into the database
        games_created = 0
        games_updated = 0
        
        for _, row in df_cleaned.iterrows():
            home_team_name = row["Home/Neutral"]
            away_team_name = row["Visitor/Neutral"]
            
            home_team_code = team_name_mapping.get(home_team_name)
            away_team_code = team_name_mapping.get(away_team_name)
            
            if not home_team_code or not away_team_code:
                self.stdout.write(self.style.WARNING(f"Could not map team name: {home_team_name} or {away_team_name}"))
                continue
                
            try:
                home_team = Team.objects.get(code=home_team_code)
                away_team = Team.objects.get(code=away_team_code)
                
                game_date = pd.to_datetime(row["Date"]).date()
                game_id = int(row["game_id"])
                
                # Determine game status
                home_score = int(row["PTS.1"]) if row["PTS.1"] != 0 else None
                away_score = int(row["PTS"]) if row["PTS"] != 0 else None
                
                if home_score is not None and away_score is not None:
                    status = 'finished'
                else:
                    status = 'scheduled'
                
                # Create or update game
                game, created = Game.objects.update_or_create(
                    game_id=game_id,
                    defaults={
                        'date': game_date,
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_score': home_score,
                        'away_score': away_score,
                        'attendance': int(row["Attend"]) if row["Attend"] != 0 else None,
                        'arena': row["Arena"] if not pd.isna(row["Arena"]) else "",
                        'status': status
                    }
                )
                
                if created:
                    games_created += 1
                else:
                    games_updated += 1
                    
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error importing game: {str(e)}"))
                logger.error(f"Error importing game: {str(e)}")
        
        self.stdout.write(self.style.SUCCESS(f"Schedule data imported: {games_created} games created, {games_updated} games updated"))
    
    def scrape_team_stats(self, save_csv=False):
        """Scrape NBA team statistics data for the current season"""
        self.stdout.write("Scraping NBA team statistics data...")
        
        # Team codes for all 30 NBA teams
        nba_teams = [
            "ATL", "BOS", "BRK", "CHO", "CHI", "CLE", "DAL", "DEN", "DET", 
            "GSW", "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", 
            "NOP", "NYK", "OKC", "ORL", "PHI", "PHO", "POR", "SAC", "SAS", 
            "TOR", "UTA", "WAS",
        ]
        
        # Base URL for team stats
        base_url = "https://www.basketball-reference.com/teams/{}/2025.html"
        
        dfs = []
        driver = self.setup_driver()
        
        for team_code in nba_teams:
            try:
                url = base_url.format(team_code)
                self.stdout.write(f"Scraping stats for {team_code}...")
                
                driver.get(url)
                time.sleep(3)  # Wait for the page to load
                
                # Wait for table to be present
                try:
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.ID, "team_and_opponent"))
                    )
                except:
                    self.stdout.write(self.style.WARNING(f"Team stats table not found for {team_code}. Skipping..."))
                    continue
                
                # Get the page source and parse with BeautifulSoup
                soup = BeautifulSoup(driver.page_source, "html.parser")
                team_table = soup.find("table", id="team_and_opponent")
                
                if team_table:
                    html_str = str(team_table)  # Convert to string
                    df = pd.read_html(StringIO(html_str))[0]  # Use StringIO to avoid FutureWarning
                    
                    row = team_table.find("tr", {"data-row": "1"})
                    if row:
                        row_data = [td.text.strip() for td in row.find_all("td")]
                        if row_data:
                            row_df = pd.DataFrame([row_data])
                            
                            # Insert team code as the first column
                            row_df.insert(0, "Team", team_code)
                            dfs.append(row_df)
                            self.stdout.write(f"Successfully scraped stats for {team_code}")
                        else:
                            self.stdout.write(self.style.WARNING(f"No data in row for {team_code}. Skipping..."))
                    else:
                        self.stdout.write(self.style.WARNING(f"No data-row='1' found for {team_code}. Skipping..."))
                else:
                    self.stdout.write(self.style.WARNING(f"No team stats table found for {team_code}. Skipping..."))
            
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error scraping {team_code} stats: {str(e)}"))
                logger.error(f"Error scraping {team_code} stats: {str(e)}")
        
        driver.quit()
        
        if not dfs:
            self.stdout.write(self.style.ERROR("No team stats data scraped"))
            return
        
        # Combine all teams' data
        teamstats_df = pd.concat(dfs, ignore_index=True)
        
        # Define the new header names
        # May need adjustment based on actual data
        new_headers = ['Team', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', 
                       '2P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 
                       'BLK', 'TOV', 'PF', 'PT']
        
        # Assign the new headers
        if len(teamstats_df.columns) == len(new_headers):
            teamstats_df.columns = new_headers
        else:
            self.stdout.write(self.style.WARNING(f"Column count mismatch: {len(teamstats_df.columns)} columns found, expected {len(new_headers)}"))
            # Add Team_id column if missing
            if 'Team_id' not in teamstats_df.columns:
                teamstats_df.insert(0, "Team_id", teamstats_df.index + 1)
        
        # Save as CSV if requested
        if save_csv:
            csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'cleaned_nba_team_stats.csv')
            teamstats_df.to_csv(csv_path, index=False)
            self.stdout.write(f"Saved team stats data to {csv_path}")
        
        # Import team stats into the database
        stats_created = 0
        stats_updated = 0
        
        for _, row in teamstats_df.iterrows():
            team_code = row["Team"]
            
            try:
                # Get team from database
                team = Team.objects.get(code=team_code)
                
                # Create mapping from dataframe columns to model fields
                stats_mapping = {
                    'mp': float(row['MP']) if 'MP' in row else 0,
                    'fg': float(row['FG']) if 'FG' in row else 0,
                    'fga': float(row['FGA']) if 'FGA' in row else 0,
                    'fg_pct': float(row['FG%'].replace('%', '')) / 100 if 'FG%' in row else 0,
                    'fg3': float(row['3P']) if '3P' in row else 0,
                    'fg3a': float(row['3PA']) if '3PA' in row else 0,
                    'fg3_pct': float(row['3P%'].replace('%', '')) / 100 if '3P%' in row else 0,
                    'fg2': float(row['2P']) if '2P' in row else 0,
                    'fg2a': float(row['2PA']) if '2PA' in row else 0,
                    'fg2_pct': float(row['2P%'].replace('%', '')) / 100 if '2P%' in row else 0,
                    'ft': float(row['FT']) if 'FT' in row else 0,
                    'fta': float(row['FTA']) if 'FTA' in row else 0,
                    'ft_pct': float(row['FT%'].replace('%', '')) / 100 if 'FT%' in row else 0,
                    'orb': float(row['ORB']) if 'ORB' in row else 0,
                    'drb': float(row['DRB']) if 'DRB' in row else 0,
                    'trb': float(row['TRB']) if 'TRB' in row else 0,
                    'ast': float(row['AST']) if 'AST' in row else 0,
                    'stl': float(row['STL']) if 'STL' in row else 0,
                    'blk': float(row['BLK']) if 'BLK' in row else 0,
                    'tov': float(row['TOV']) if 'TOV' in row else 0,
                    'pf': float(row['PF']) if 'PF' in row else 0,
                    'pts': float(row['PT']) if 'PT' in row else 0,
                }
                
                # Create or update team stats
                team_stats, created = TeamStats.objects.update_or_create(
                    team=team,
                    season="2024-2025",
                    defaults=stats_mapping
                )
                
                if created:
                    stats_created += 1
                else:
                    stats_updated += 1
                    
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error importing stats for {team_code}: {str(e)}"))
                logger.error(f"Error importing stats for {team_code}: {str(e)}")
        
        self.stdout.write(self.style.SUCCESS(f"Team stats data imported: {stats_created} created, {stats_updated} updated"))
    
    def scrape_standings(self, save_csv=False):
        """Scrape NBA standings data for the current season"""
        self.stdout.write("Scraping NBA standings data...")
        
        driver = self.setup_driver()
        
        try:
            # URL for standings
            url = "https://www.basketball-reference.com/leagues/NBA_2025_standings.html"
            driver.get(url)
            time.sleep(5)  # Wait for page to load
            
            # Wait for tables to be loaded
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "confs_standings_E")))
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "confs_standings_W")))
            
            # Get page source
            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")
            
            # Find the correct tables
            standingsE_table = soup.find(id="confs_standings_E")
            standingsW_table = soup.find(id="confs_standings_W")
            
            if standingsE_table and standingsW_table:
                standingsE = pd.read_html(str(standingsE_table))[0]
                standingsW = pd.read_html(str(standingsW_table))[0]
                
                # Save as CSV if requested
                if save_csv:
                    csv_path_e = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'standingsE.csv')
                    csv_path_w = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'standingsW.csv')
                    standingsE.to_csv(csv_path_e, index=False)
                    standingsW.to_csv(csv_path_w, index=False)
                    self.stdout.write(f"Saved standings data to {csv_path_e} and {csv_path_w}")
                
                self.stdout.write(self.style.SUCCESS("Standings data scraped successfully"))
            else:
                self.stdout.write(self.style.ERROR("Error: No standings tables found."))
        
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error scraping standings: {str(e)}"))
            logger.error(f"Error scraping standings: {str(e)}")
        
        finally:
            driver.quit()