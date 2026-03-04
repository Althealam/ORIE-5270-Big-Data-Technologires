from collections import OrderedDict
from typing import Tuple, Optional
import math

## TODO: Implement other classes or helper functions if needed

# =========================
# Main mechanism
# =========================
class CachingMechanism(object):
    """A Python emulator of a caching mechanism for movies (Video cache++)."""

    def __init__(self, movie_list, cache_list, movies_per_cache, ttl: int):
        """
        Parameters
        ----------
        movie_list : list[str]
            Valid movie titles.
        cache_list : list[tuple[str, float, float]]
            Each tuple is (LocationName, X, Y).
        movies_per_cache : int
            Capacity K per cache.
        ttl : int
            TTL for new inserts; expiration time is t + ttl.
        """
        self.movie_list = movie_list
        self.cache_locations = cache_list  # List of (LocationName, X, Y)
        self.K = movies_per_cache  # Maximum capacity per cache
        self.TTL = ttl

        # Initialize empty caches for each location
        # Each cache is an OrderedDict mapping movie_title -> expiration_time
        # OrderedDict maintains insertion order, enabling efficient LRU operations
        self.caches = {location[0]: OrderedDict() for location in cache_list}

    def find_nearest_cache(self, x: float, y: float) -> str:
        """
        Return nearest cache location name using Euclidean distance.
        Tie-break: lexicographically smallest LocationName.

        Time complexity: O(M) where M is the number of cache locations.
        Can be optimized to O(log M) using spatial data structures like KD-tree.
        """
        min_distance = float('inf')
        location_name = None

        for loc_name, cx, cy in self.cache_locations:
            # Calculate Euclidean distance
            distance = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)

            # Update nearest location with tie-breaking rule:
            # If distances are equal, choose lexicographically smallest name
            if distance < min_distance or (distance == min_distance and
                                          (location_name is None or loc_name < location_name)):
                min_distance = distance
                location_name = loc_name

        return location_name

    def update_cache_state(self, location_name: str, movie_title: str, t: int) -> None:
        """
        Bring a movie into the specified cache at time t.
        (This is called on misses/expired, per the prompt.)

        Implementation strategy:
        1. Remove all expired items first (they don't count toward capacity)
        2. If cache is still full, evict LRU (least recently used) valid items
        3. Insert new movie with expiration time t + TTL

        Time complexity: O(log K) amortized
        - Removing expired items: O(K) worst case, but O(1) amortized
          (each item is expired and removed at most once)
        - Eviction: O(1) using OrderedDict
        - Insertion: O(1)
        """
        cache = self.caches[location_name]

        # Step 1: Remove all expired items
        # Expired items do not count toward cache capacity
        expired_keys = [key for key, exp_time in cache.items() if t >= exp_time]
        for key in expired_keys:
            del cache[key]

        # Step 2: If cache is still full (only counting valid items), evict LRU items
        while len(cache) >= self.K:
            # OrderedDict maintains insertion/access order
            # The first item is the least recently used (LRU)
            oldest_key = next(iter(cache))
            del cache[oldest_key]

        # Step 3: Insert new movie with expiration time
        # New items are automatically added to the end of OrderedDict (most recently used)
        cache[movie_title] = t + self.TTL

    def lookup(self, movie_title: str, user_x: float, user_y: float, t: int) -> Tuple[bool, Optional[str]]:
        """
        Look up a movie given title and user coordinates at time t.

        If nearest cache contains a valid copy, return (True, LocationName) and update recency.
        Otherwise return (False, None) and insert/refresh in nearest cache with expiration t+TTL.

        Implementation follows the three-step process:
        Step 1: Find nearest cache using Euclidean distance
        Step 2: Check if movie exists and is valid (not expired)
                - If hit: update LRU order (true LRU: update on every access)
                - If miss/expired: proceed to Step 3
        Step 3: Update cache state with new entry

        Time complexity: O(log M) + O(log K) = O(log(M+K))
        - find_nearest_cache: O(M) (can be O(log M) with spatial indexing)
        - Cache lookup: O(1) with OrderedDict
        - move_to_end: O(1)
        - update_cache_state: O(log K) amortized
        """
        # Step 1: Find nearest cache location
        location_name = self.find_nearest_cache(user_x, user_y)
        cache = self.caches[location_name]

        # Step 2: Check if movie is in cache and valid
        if movie_title in cache:
            expiration_time = cache[movie_title]

            # Check if not expired (valid if t < expiration_time)
            if t < expiration_time:
                # Cache hit! Update LRU order
                # True LRU: recency is updated on every access, not just insertion
                cache.move_to_end(movie_title)
                return (True, location_name)
            else:
                # Movie is expired, remove it
                del cache[movie_title]

        # Step 3: Cache miss or expired - update cache state
        self.update_cache_state(location_name, movie_title, t)
        return (False, None)
