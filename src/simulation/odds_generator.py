"""
Odds Generator for Season Simulation Engine.

Generates realistic betting odds based on historical patterns and team performance.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class OddsGenerator:
    """
    Generates realistic betting odds for match simulation.

    Features:
    - Historical odds patterns analysis
    - Team strength-based odds generation
    - Market margin simulation
    - Realistic odds distributions
    """

    def __init__(self, training_data_path: str):
        """Initialize with historical training data."""
        self.training_data_path = Path(training_data_path)
        self.training_data = self._load_training_data()
        self.team_strengths = self._calculate_team_strengths()
        self.odds_distributions = self._analyze_odds_distributions()

        logger.info(f"OddsGenerator initialized with {len(self.training_data)} historical matches")
        logger.info(f"Team strengths calculated for {len(self.team_strengths)} teams")

    def _load_training_data(self) -> pd.DataFrame:
        """Load historical training data."""
        if not self.training_data_path.exists():
            raise FileNotFoundError(f"Training data not found: {self.training_data_path}")

        data = pd.read_parquet(self.training_data_path)

        # Ensure we have required odds columns
        required_cols = ["B365H", "B365D", "B365A", "HomeTeam", "AwayTeam", "FTR"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return data

    def _calculate_team_strengths(self) -> dict[str, dict[str, float]]:
        """Calculate team strengths based on historical performance."""
        team_stats = {}

        # Get all unique teams
        all_teams = set(self.training_data["HomeTeam"].unique()) | set(
            self.training_data["AwayTeam"].unique()
        )

        for team in all_teams:
            # Home performance
            home_matches = self.training_data[self.training_data["HomeTeam"] == team]
            home_wins = (home_matches["FTR"] == "H").sum()
            home_draws = (home_matches["FTR"] == "D").sum()
            home_losses = (home_matches["FTR"] == "A").sum()
            home_total = len(home_matches)

            # Away performance
            away_matches = self.training_data[self.training_data["AwayTeam"] == team]
            away_wins = (away_matches["FTR"] == "A").sum()
            away_draws = (away_matches["FTR"] == "D").sum()
            away_losses = (away_matches["FTR"] == "H").sum()
            away_total = len(away_matches)

            # Calculate strength metrics
            total_matches = home_total + away_total
            if total_matches > 0:
                overall_win_rate = (home_wins + away_wins) / total_matches
                home_win_rate = home_wins / home_total if home_total > 0 else 0.5
                away_win_rate = away_wins / away_total if away_total > 0 else 0.3
                draw_rate = (home_draws + away_draws) / total_matches

                # Calculate average odds when this team was involved
                team_matches = pd.concat(
                    [
                        home_matches[["B365H", "B365D", "B365A"]].rename(
                            columns={
                                "B365H": "team_odds",
                                "B365D": "draw_odds",
                                "B365A": "opponent_odds",
                            }
                        ),
                        away_matches[["B365A", "B365D", "B365H"]].rename(
                            columns={
                                "B365A": "team_odds",
                                "B365D": "draw_odds",
                                "B365H": "opponent_odds",
                            }
                        ),
                    ]
                )

                avg_team_odds = team_matches["team_odds"].mean()

                team_stats[team] = {
                    "overall_win_rate": overall_win_rate,
                    "home_win_rate": home_win_rate,
                    "away_win_rate": away_win_rate,
                    "draw_rate": draw_rate,
                    "avg_odds": avg_team_odds,
                    "strength_score": overall_win_rate + (1 / avg_team_odds) * 0.3,
                    "total_matches": total_matches,
                }
            else:
                # Default for teams with no history
                team_stats[team] = {
                    "overall_win_rate": 0.4,
                    "home_win_rate": 0.5,
                    "away_win_rate": 0.3,
                    "draw_rate": 0.3,
                    "avg_odds": 3.0,
                    "strength_score": 0.4,
                    "total_matches": 0,
                }

        return team_stats

    def _analyze_odds_distributions(self) -> dict[str, dict[str, float]]:
        """Analyze historical odds distributions."""
        # Remove rows with missing odds
        clean_data = self.training_data.dropna(subset=["B365H", "B365D", "B365A"])

        distributions = {
            "home_odds": {
                "mean": clean_data["B365H"].mean(),
                "std": clean_data["B365H"].std(),
                "min": clean_data["B365H"].min(),
                "max": clean_data["B365H"].max(),
                "p25": clean_data["B365H"].quantile(0.25),
                "p75": clean_data["B365H"].quantile(0.75),
            },
            "draw_odds": {
                "mean": clean_data["B365D"].mean(),
                "std": clean_data["B365D"].std(),
                "min": clean_data["B365D"].min(),
                "max": clean_data["B365D"].max(),
                "p25": clean_data["B365D"].quantile(0.25),
                "p75": clean_data["B365D"].quantile(0.75),
            },
            "away_odds": {
                "mean": clean_data["B365A"].mean(),
                "std": clean_data["B365A"].std(),
                "min": clean_data["B365A"].min(),
                "max": clean_data["B365A"].max(),
                "p25": clean_data["B365A"].quantile(0.25),
                "p75": clean_data["B365A"].quantile(0.75),
            },
        }

        return distributions

    def generate_odds(
        self, home_team: str, away_team: str, add_noise: bool = True
    ) -> tuple[float, float, float]:
        """
        Generate realistic odds for a match.

        Args:
            home_team: Home team name
            away_team: Away team name
            add_noise: Whether to add realistic noise to odds

        Returns:
            Tuple of (home_odds, draw_odds, away_odds)
        """
        # Get team strengths
        home_stats = self.team_strengths.get(home_team, self.team_strengths.get("Arsenal", {}))
        away_stats = self.team_strengths.get(away_team, self.team_strengths.get("Chelsea", {}))

        # Calculate implied probabilities based on team strengths
        home_strength = home_stats.get("strength_score", 0.5)
        away_strength = away_stats.get("strength_score", 0.5)

        # Adjust for home advantage (historically ~0.1 probability boost)
        home_advantage = 0.1
        home_prob_base = home_strength / (home_strength + away_strength) + home_advantage
        away_prob_base = away_strength / (home_strength + away_strength)

        # Ensure probabilities are reasonable
        home_prob_base = max(0.2, min(0.7, home_prob_base))
        away_prob_base = max(0.2, min(0.7, away_prob_base))

        # Draw probability (typically 20-35% in Premier League)
        draw_prob_base = max(0.2, min(0.35, 1 - home_prob_base - away_prob_base + 0.25))

        # Normalize probabilities
        total_prob = home_prob_base + draw_prob_base + away_prob_base
        home_prob = home_prob_base / total_prob
        draw_prob = draw_prob_base / total_prob
        away_prob = away_prob_base / total_prob

        # Convert to odds (with bookmaker margin)
        margin = np.random.normal(0.05, 0.01)  # Typical 5% margin
        margin = max(0.02, min(0.08, margin))  # Clamp margin

        home_odds = (1 + margin) / home_prob
        draw_odds = (1 + margin) / draw_prob
        away_odds = (1 + margin) / away_prob

        # Add realistic noise if requested
        if add_noise:
            noise_factor = 0.05  # 5% noise
            home_odds *= np.random.normal(1.0, noise_factor)
            draw_odds *= np.random.normal(1.0, noise_factor)
            away_odds *= np.random.normal(1.0, noise_factor)

        # Ensure odds are within realistic bounds
        home_odds = max(1.1, min(15.0, home_odds))
        draw_odds = max(2.5, min(6.0, draw_odds))
        away_odds = max(1.1, min(15.0, away_odds))

        # Round to typical betting precision
        home_odds = round(home_odds, 2)
        draw_odds = round(draw_odds, 2)
        away_odds = round(away_odds, 2)

        return home_odds, draw_odds, away_odds

    def generate_odds_for_matches(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Generate odds for multiple matches."""
        enriched_matches = matches.copy()

        # Generate odds for each match
        home_odds = []
        draw_odds = []
        away_odds = []

        for _, match in matches.iterrows():
            h_odds, d_odds, a_odds = self.generate_odds(match["HomeTeam"], match["AwayTeam"])
            home_odds.append(h_odds)
            draw_odds.append(d_odds)
            away_odds.append(a_odds)

        # Add generated odds to dataframe
        enriched_matches["generated_home_odds"] = home_odds
        enriched_matches["generated_draw_odds"] = draw_odds
        enriched_matches["generated_away_odds"] = away_odds

        # Also calculate margin-adjusted probabilities for model features
        enriched_matches["home_prob_margin_adj"] = 1 / enriched_matches["generated_home_odds"]
        enriched_matches["draw_prob_margin_adj"] = 1 / enriched_matches["generated_draw_odds"]
        enriched_matches["away_prob_margin_adj"] = 1 / enriched_matches["generated_away_odds"]

        # Normalize probabilities to sum to 1 (remove bookmaker margin)
        total_prob = (
            enriched_matches["home_prob_margin_adj"]
            + enriched_matches["draw_prob_margin_adj"]
            + enriched_matches["away_prob_margin_adj"]
        )

        enriched_matches["home_prob_margin_adj"] /= total_prob
        enriched_matches["draw_prob_margin_adj"] /= total_prob
        enriched_matches["away_prob_margin_adj"] /= total_prob

        logger.info(f"Generated odds for {len(matches)} matches")
        return enriched_matches

    def get_team_strength_summary(self) -> pd.DataFrame:
        """Get summary of all team strengths."""
        summary_data = []
        for team, stats in self.team_strengths.items():
            summary_data.append(
                {
                    "team": team,
                    "strength_score": stats["strength_score"],
                    "overall_win_rate": stats["overall_win_rate"],
                    "home_win_rate": stats["home_win_rate"],
                    "away_win_rate": stats["away_win_rate"],
                    "avg_odds": stats["avg_odds"],
                    "total_matches": stats["total_matches"],
                }
            )

        return pd.DataFrame(summary_data).sort_values("strength_score", ascending=False)

    def validate_odds_realism(self, generated_odds: pd.DataFrame) -> dict[str, float]:
        """Validate that generated odds are realistic compared to historical data."""
        historical = self.training_data.dropna(subset=["B365H", "B365D", "B365A"])

        # Compare distributions
        validation = {
            "home_odds_mean_diff": abs(
                generated_odds["generated_home_odds"].mean() - historical["B365H"].mean()
            ),
            "draw_odds_mean_diff": abs(
                generated_odds["generated_draw_odds"].mean() - historical["B365D"].mean()
            ),
            "away_odds_mean_diff": abs(
                generated_odds["generated_away_odds"].mean() - historical["B365A"].mean()
            ),
            "realism_score": 0.0,
        }

        # Calculate overall realism score (lower is better)
        avg_diff = (
            validation["home_odds_mean_diff"]
            + validation["draw_odds_mean_diff"]
            + validation["away_odds_mean_diff"]
        ) / 3

        validation["realism_score"] = 1 / (1 + avg_diff)  # Score between 0 and 1

        return validation
