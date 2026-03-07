# requirement:
# 1. space: O(k)
# 2. time: O(logk)
# 3. not allowed to store the entire bid history

import heapq
class Auctioneer(object):
    """A class emulating an auctioneer that tracks the highest K bids."""

    def __init__(self, K):
        self.K = K
        # TODO: Initialize any data structures or other fields necessary here.
        self.min_heap = [] # space: O(k)

    def process_next_bid(self, bid_value):
        """Process the next bid, updating the highest K bids if necessary.

        This method should return `True` if it caused an update to the list
        of K bids and `False` if not.
        """
        # TODO: Fill in your code here.
        if len(self.min_heap)<self.K: # time: O(logk)
            heapq.heappush(self.min_heap, bid_value)
            return True
        else:
            if bid_value>self.min_heap[0]:
                heapq.heappop(self.min_heap) # time: O(logk)
                heapq.heappush(self.min_heap, bid_value) # time: O(logk)
                return True
        return False


    def get_bids(self):
        """Return the highest K bids encountered so far in sorted order.

        This operation should not modify the data structure used to track the
        highest K bids. The highest K bids should be returned in increasing
        order.
        """
        # TODO: Fill in your code here.
        return self.sorted(self.min_heap) # time: O(logk)
