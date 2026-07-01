from quantforge.research.leaderboard_engine import (
    LeaderboardEngine,
)

lb = LeaderboardEngine()

print("=" * 80)
print("CHAMPION")
print("=" * 80)

print(lb.champion())

print()

print("=" * 80)
print("TOP 10")
print("=" * 80)

print(lb.top())
