"""Test script for caching_mechanism.py"""

from caching_mechanism import CachingMechanism

# Test setup
movie_list = ["Movie1", "Movie2", "Movie3", "Movie4"]
cache_list = [
    ("CacheA", 0.0, 0.0),
    ("CacheB", 10.0, 10.0),
    ("CacheC", 5.0, 5.0)
]
K = 3  # Max 3 movies per cache
TTL = 10  # Movies expire after 10 time units

mechanism = CachingMechanism(movie_list, cache_list, K, TTL)

print("=" * 60)
print("Test 1: First lookup (cache miss)")
print("=" * 60)
result = mechanism.lookup("Movie1", 1.0, 1.0, 0)
print(f"lookup('Movie1', (1,1), t=0) -> {result}")
print(f"Expected: (False, None)")
print()

print("=" * 60)
print("Test 2: Second lookup (cache hit)")
print("=" * 60)
result = mechanism.lookup("Movie1", 1.0, 1.0, 1)
print(f"lookup('Movie1', (1,1), t=1) -> {result}")
print(f"Expected: (True, 'CacheA')")
print()

print("=" * 60)
print("Test 3: Lookup after expiration")
print("=" * 60)
result = mechanism.lookup("Movie1", 1.0, 1.0, 11)
print(f"lookup('Movie1', (1,1), t=11) -> {result}")
print(f"Expected: (False, None) - expired and re-cached")
print()

print("=" * 60)
print("Test 4: Fill cache to capacity")
print("=" * 60)
mechanism.lookup("Movie2", 1.0, 1.0, 12)
mechanism.lookup("Movie3", 1.0, 1.0, 13)
print(f"Cache CacheA should now have: Movie1, Movie2, Movie3")
print()

print("=" * 60)
print("Test 5: LRU eviction")
print("=" * 60)
# Access Movie2 and Movie3 to make Movie1 the LRU
mechanism.lookup("Movie2", 1.0, 1.0, 14)
mechanism.lookup("Movie3", 1.0, 1.0, 15)
# Now insert Movie4, should evict Movie1
mechanism.lookup("Movie4", 1.0, 1.0, 16)
print("Inserted Movie4, should have evicted Movie1 (LRU)")
print()

print("=" * 60)
print("Test 6: Verify eviction")
print("=" * 60)
result = mechanism.lookup("Movie1", 1.0, 1.0, 17)
print(f"lookup('Movie1', (1,1), t=17) -> {result}")
print(f"Expected: (False, None) - Movie1 was evicted")
print()

print("=" * 60)
print("Test 7: Nearest cache selection")
print("=" * 60)
nearest = mechanism.find_nearest_cache(4.5, 4.5)
print(f"Nearest cache to (4.5, 4.5) -> {nearest}")
print(f"Expected: 'CacheC' (at 5,5)")
print()

print("=" * 60)
print("Test 8: Tie-breaking with lexicographical order")
print("=" * 60)
# Add two caches at same distance
mechanism_tie = CachingMechanism(
    movie_list,
    [("CacheZ", 0.0, 0.0), ("CacheA", 0.0, 0.0)],
    K, TTL
)
nearest = mechanism_tie.find_nearest_cache(0.0, 0.0)
print(f"Nearest cache when tied -> {nearest}")
print(f"Expected: 'CacheA' (lexicographically first)")
print()

print("=" * 60)
print("Final cache states:")
print("=" * 60)
for loc_name, cache in mechanism.caches.items():
    print(f"{loc_name}: {dict(cache)}")
