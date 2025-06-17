# scraper/views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.contrib.auth.decorators import user_passes_test
from django.core.management import call_command
import subprocess
import os
import json
from datetime import datetime
from predictor_api.models import Team, TeamStats, Game, Prediction

def is_staff(user):
    """Check if user is staff"""
    return user.is_staff

def scraper_home(request):
    """Home page for the scraper app"""
    context = {
        'title': 'NBA Data Scraper',
        'last_update': get_last_update(),
    }
    return render(request, 'scraper/home.html', context)

@user_passes_test(is_staff)
def run_scraper(request):
    """Run the scraper as a background task"""
    if request.method == 'POST':
        try:
            # Parse request data
            data = json.loads(request.body)
            scrape_schedule = data.get('schedule', False)
            scrape_standings = data.get('standings', False)
            scrape_stats = data.get('stats', False)
            save_csv = data.get('save_csv', False)
            
            # Build command arguments
            args = []
            if scrape_schedule:
                args.append('--schedule')
            if scrape_standings:
                args.append('--standings')
            if scrape_stats:
                args.append('--stats')
            if not args:  # If no specific items, scrape all
                args.append('--all')
            if save_csv:
                args.append('--save-csv')
            
            # Run the management command asynchronously
            subprocess.Popen(['python', 'manage.py', 'scrape_nba_data'] + args)
            
            # Update last update timestamp
            update_last_update()
            
            return JsonResponse({'status': 'success', 'message': 'Scraper started'})
        
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    
    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)

def data_status(request):
    """Get the current status of the data in the database"""
    try:
        team_count = Team.objects.count()
        stats_count = TeamStats.objects.count()
        games_count = Game.objects.count()
        predictions_count = Prediction.objects.count()
        
        # Get counts of different game statuses
        scheduled_games = Game.objects.filter(status='scheduled').count()
        finished_games = Game.objects.filter(status='finished').count()
        
        # Get timestamp of last update
        last_update = get_last_update()
        
        return JsonResponse({
            'status': 'success',
            'data': {
                'teams': team_count,
                'team_stats': stats_count,
                'games': games_count,
                'predictions': predictions_count,
                'scheduled_games': scheduled_games,
                'finished_games': finished_games,
                'last_update': last_update,
            }
        })
    
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

def get_last_update():
    """Get the timestamp of the last data update"""
    try:
        status_file = os.path.join(os.path.dirname(__file__), 'data', 'status.json')
        
        if not os.path.exists(status_file):
            return None
            
        with open(status_file, 'r') as f:
            status = json.load(f)
            return status.get('last_update', None)
    
    except Exception:
        return None

def update_last_update():
    """Update the timestamp of the last data update"""
    try:
        status_file = os.path.join(os.path.dirname(__file__), 'data', 'status.json')
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(status_file), exist_ok=True)
        
        status = {}
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                status = json.load(f)
                
        status['last_update'] = datetime.now().isoformat()
        
        with open(status_file, 'w') as f:
            json.dump(status, f)
    
    except Exception:
        pass