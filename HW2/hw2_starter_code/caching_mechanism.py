## TODO: Implement other classes or helper functions if needed

# =========================
# Main mechanism
# =========================
from collections import defaultdict, OrderedDict
from typing import Tuple, Optional

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
        ## TODO: Implement it
        self.movies = set(movie_list)
        self.cache_list = cache_list # list of (locationname, x, y)
        self.ttl = ttl
        self.movies_per_cache = movies_per_cache # maximum capacity per cache

        # as the questions mentioned, each cache locations maintains K cache, which can store at most K movies
        # each cache is an orderdict mapping movie_title->expiration_time
        self.caches = {}
        for location_name, x, y in cache_list:
            self.caches[location_name] = {
                'items': {}, # movie->expiration
                # items={"A":100, "B": 200, "C": 300}
                'lru': OrderedDict() # MovieA, MovieB, MovieC
            }

    def find_nearest_cache(self, x: float, y: float) -> str:
        """
        Return nearest cache location name using Euclidean distance.
        Tie-break: lexicographically smallest LocationName.
        """
        ## TODO: Implement it
        location_name = None
        min_distance = float('inf')
        for location, xi, yi in self.cache_list:
            current_distance = (x-xi)**2+(y-yi)**2
            # NOTE: need to compare their lexicograph when their distance are the same
            if current_distance<min_distance or (current_distance==min_distance and location<location_name):
                min_distance = current_distance
                location_name = location
        return location_name

    def update_cache_state(self, location_name: str, movie_title: str, t: int) -> None:
        """
        Bring a movie into the specified cache at time t.
        (This is called on misses/expired, per the prompt.)

        """
        ## TODO: Implement it
        cache = self.caches[location_name]
        items = cache['items'] # movie->expiration
        lru = cache['lru'] # main recency

        # 1. clean expired movies in this cache
        expired = []
        for movie, expiration in items.items(): # movie, expiration_time
            if expiration<=t:
                expired.append(movie)
        for movie in expired:
            del items[movie]
            if movie in lru:
                del lru[movie]
        
        expiration_time = t+self.ttl

        # 2. movie already exits
        if movie_title in items:
            items[movie_title] = expiration_time
            lru.move_to_end(movie_title)
            return 
        
        # 3. cache full -> evict LRU
        if len(items)>=self.movies_per_cache:
            oldest_movie, _ = lru.popitem(last=False)
            del items[oldest_movie]
        
        # 4. insert movie
        items[movie_title] = expiration_time
        lru[movie_title] = None



    def lookup(self, movie_title: str, user_x: float, user_y: float, t: int) -> Tuple[bool, Optional[str]]:
        """
        Look up a movie given title and user coordinates at time t.

        If nearest cache contains a valid copy, return (True, LocationName) and update recency.
        Otherwise return (False, None) and insert/refresh in nearest cache with expiration t+TTL.
        """
        ## TODO: Implement it
        # 1. find the nearest location
        nearest_location = self.find_nearest_cache(user_x, user_y)

        # 2. get the cache for this location, which include [items, lru, expire_heap]
        cache = self.caches[nearest_location]
        items = cache['items'] # movie->expiration_time
        lru = cache['lru']

        # 3. clean expired movies
        expired = []
        for movie, expiration_time in items.items():
            if expiration_time<=t:
                expired.append(movie)
        
        for movie in expired:
            del items[movie]
            if movie in lru:
                del lru[movie]

        # 4. cache hit
        if movie_title in items and items[movie_title]>t:
            lru.move_to_end(movie_title)
            return (True, nearest_location)

        # 5. cache miss
        self.update_cache_state(nearest_location, movie_title, t)
        return (False, None)


# movie_list = ["Movie1", "Movie2", "Movie3", "Movie4"]
# # directions and their x and y
# cache_list = [
#     ("CacheA", 0.0, 0.0),
#     ("CacheB", 10.0, 10.0),
#     ("CacheC", 5.0, 5.0)
# ]
# movies_per_cache = 3  # Max 3 movies per cache
# TTL = 10  # Movies expire after 10 time units
# caching_mechanism = CachingMechanism(movie_list, cache_list, movies_per_cache, TTL)
# print(caching_mechanism.caches)


# movie_title = "Movie1"
# str_x = 2
# str_y = 3
# nearest_location = caching_mechanism.find_nearest_cache(str_x, str_y)
# print(nearest_location)

# caching_mechanism.lookup('A', 1, 1, 1)
# caching_mechanism.lookup('A', 1, 1, 2)
# caching_mechanism.lookup('B', 1, 1, 3)
# caching_mechanism.lookup('C', 1, 1, 4)
# print(caching_mechanism.caches)