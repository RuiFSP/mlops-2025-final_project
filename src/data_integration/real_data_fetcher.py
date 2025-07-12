"""
Real data fetcher for Premier League matches using football-data.org API.
"""

import logging
import os
import time
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class RealDataFetcher:
    """Fetches real Premier League data from football-data.org API."""

    def __init__(self):
        """Initialize the real data fetcher."""
        self.api_key = os.getenv("FOOTBALL_DATA_API_KEY")
        self.base_url = "https://api.football-data.org/v4"
        self.headers = {"X-Auth-Token": self.api_key if self.api_key else ""}

        # Premier League ID in football-data.org
        self.premier_league_id = 2021

        # Fallback team mapping for consistency
        self.team_mapping = {
            "Manchester City FC": "Manchester City",
            "Manchester United FC": "Manchester United",
            "Liverpool FC": "Liverpool",
            "Arsenal FC": "Arsenal",
            "Chelsea FC": "Chelsea",
            "Tottenham Hotspur FC": "Tottenham",
            "Newcastle United FC": "Newcastle",
            "Brighton & Hove Albion FC": "Brighton",
            "Aston Villa FC": "Aston Villa",
            "West Ham United FC": "West Ham",
            "Brentford FC": "Brentford",
            "Fulham FC": "Fulham",
            "Crystal Palace FC": "Crystal Palace",
            "Wolverhampton Wanderers FC": "Wolves",
            "Everton FC": "Everton",
            "Nottingham Forest FC": "Nottingham Forest",
            "Leicester City FC": "Leicester",
            "Southampton FC": "Southampton",
            "AFC Bournemouth": "Bournemouth",
            "Sheffield United FC": "Sheffield United",
            "Burnley FC": "Burnley",
            "Luton Town FC": "Luton",
        }

    def _make_request(self, endpoint: str, params: dict | None = None) -> dict | None:
        """Make a request to the football-data.org API."""
        try:
            url = f"{self.base_url}/{endpoint}"
            response = requests.get(url, headers=self.headers, params=params)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                logger.warning("Rate limit exceeded. Waiting 60 seconds...")
                time.sleep(60)
                return self._make_request(endpoint, params)
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error making API request: {e}")
            return None

    def _normalize_team_name(self, team_name: str) -> str:
        """Normalize team names to match our training data format."""
        return self.team_mapping.get(team_name, team_name)

    def get_upcoming_matches(self, days_ahead: int = 7) -> list[dict]:
        """Get upcoming Premier League matches."""
        logger.info(f"Fetching upcoming matches for next {days_ahead} days...")

        # If no API key, fall back to simulated matches
        if not self.api_key:
            logger.warning("No API key provided. Using simulated matches as fallback.")
            return self._generate_fallback_matches()

        # Get current season matches
        today = datetime.now()
        date_to = (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

        params = {"dateFrom": today.strftime("%Y-%m-%d"), "dateTo": date_to, "status": "SCHEDULED"}

        data = self._make_request(f"competitions/{self.premier_league_id}/matches", params)

        if not data or "matches" not in data:
            logger.warning("No upcoming matches found from API. Using fallback matches.")
            return self._generate_fallback_matches()

        matches = []
        for match in data["matches"]:
            try:
                # Extract match information
                home_team = self._normalize_team_name(match["homeTeam"]["name"])
                away_team = self._normalize_team_name(match["awayTeam"]["name"])
                match_date = datetime.fromisoformat(match["utcDate"].replace("Z", "+00:00"))

                # Generate match ID
                match_id = f"real_{match_date.strftime('%Y%m%d')}_{match['id']}"

                # Get odds (if available, otherwise use realistic defaults)
                home_odds = self._get_realistic_odds(home_team, away_team, "home")
                draw_odds = self._get_realistic_odds(home_team, away_team, "draw")
                away_odds = self._get_realistic_odds(home_team, away_team, "away")

                match_info = {
                    "match_id": match_id,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_odds": home_odds,
                    "draw_odds": draw_odds,
                    "away_odds": away_odds,
                    "match_date": match_date.date(),
                    "season": "2024/25",
                }

                matches.append(match_info)
                logger.info(f"Added real match: {home_team} vs {away_team} on {match_date.date()}")

            except Exception as e:
                logger.error(f"Error processing match: {e}")
                continue

        if not matches:
            logger.warning("No valid matches processed. Using fallback matches.")
            return self._generate_fallback_matches()

        logger.info(f"Successfully fetched {len(matches)} real upcoming matches")
        return matches

    def _get_realistic_odds(self, home_team: str, away_team: str, outcome: str) -> float:
        """Generate realistic odds based on team strength."""
        # Team strength ratings (approximate)
        team_strength = {
            "Manchester City": 0.95,
            "Arsenal": 0.88,
            "Liverpool": 0.85,
            "Chelsea": 0.80,
            "Manchester United": 0.78,
            "Tottenham": 0.75,
            "Newcastle": 0.72,
            "Brighton": 0.68,
            "Aston Villa": 0.65,
            "West Ham": 0.62,
            "Brentford": 0.58,
            "Fulham": 0.55,
            "Crystal Palace": 0.52,
            "Wolves": 0.50,
            "Everton": 0.48,
            "Nottingham Forest": 0.45,
            "Leicester": 0.42,
            "Southampton": 0.40,
            "Bournemouth": 0.38,
            "Sheffield United": 0.35,
            "Burnley": 0.32,
            "Luton": 0.30,
        }

        home_strength = team_strength.get(home_team, 0.50)
        away_strength = team_strength.get(away_team, 0.50)

        # Home advantage factor
        home_advantage = 0.1

        # Calculate probabilities
        home_prob = home_strength + home_advantage
        away_prob = away_strength
        draw_prob = 0.25  # Base draw probability

        # Normalize probabilities
        total_prob = home_prob + away_prob + draw_prob
        home_prob /= total_prob
        away_prob /= total_prob
        draw_prob /= total_prob

        # Convert to odds (with bookmaker margin)
        margin = 0.05  # 5% bookmaker margin

        if outcome == "home":
            return round(1 / (home_prob * (1 - margin)), 2)
        elif outcome == "draw":
            return round(1 / (draw_prob * (1 - margin)), 2)
        else:  # away
            return round(1 / (away_prob * (1 - margin)), 2)

    def _generate_fallback_matches(self) -> list[dict]:
        """Generate fallback matches when API is unavailable."""
        logger.info("Generating fallback matches...")

        # Use realistic Premier League teams
        teams = [
            "Arsenal",
            "Manchester City",
            "Liverpool",
            "Chelsea",
            "Manchester United",
            "Newcastle",
            "Tottenham",
            "Brighton",
            "Aston Villa",
            "West Ham",
            "Brentford",
            "Fulham",
            "Crystal Palace",
            "Wolves",
            "Everton",
        ]

        import random

        matches = []
        today = datetime.now()

        for i in range(3):  # Generate 3 realistic matches
            home_team = random.choice(teams)
            away_team = random.choice([t for t in teams if t != home_team])

            # Generate realistic odds
            home_odds = self._get_realistic_odds(home_team, away_team, "home")
            draw_odds = self._get_realistic_odds(home_team, away_team, "draw")
            away_odds = self._get_realistic_odds(home_team, away_team, "away")

            match = {
                "match_id": f"fallback_{today.strftime('%Y%m%d')}_{i}",
                "home_team": home_team,
                "away_team": away_team,
                "home_odds": home_odds,
                "draw_odds": draw_odds,
                "away_odds": away_odds,
                "match_date": today.date(),
                "season": "2024/25",
            }
            matches.append(match)

        logger.info(f"Generated {len(matches)} fallback matches")
        return matches

    def get_team_stats(self, team_name: str) -> dict:
        """Get team statistics for better prediction features."""
        # This would fetch team stats from the API
        # For now, return defaults
        return {
            "recent_form": 0.5,
            "goals_scored_avg": 1.5,
            "goals_conceded_avg": 1.2,
            "shots_per_game": 12,
            "shots_on_target_per_game": 5,
        }


if __name__ == "__main__":
    # Test the fetcher
    logging.basicConfig(level=logging.INFO)
    fetcher = RealDataFetcher()
    matches = fetcher.get_upcoming_matches(days_ahead=7)

    print(f"Fetched {len(matches)} upcoming matches:")
    for match in matches:
        print(f"  {match['home_team']} vs {match['away_team']} - {match['match_date']}")
