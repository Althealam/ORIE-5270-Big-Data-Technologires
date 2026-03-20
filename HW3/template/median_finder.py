class MedianFinder:
    """
    A data structure that supports efficiently computing the median of a
    dynamically growing stream of numbers.

    The structure supports two operations:
        1. addNum(num): add a number from the data stream
        2. findMedian(): return the current median of all inserted numbers
    """

    def __init__(self):
        """
        Initialize the data structure.
        """
        import heapq
        self.max_heap = []  # Left half (smaller elements), max heap
        self.min_heap = []  # Right half (larger elements), min heap

    def addNum(self, num) -> None:
        """
        Add a new number to the data stream.

        Parameters
        ----------
        num : int
            The number to be added to the structure.
        """
        import heapq

        # Add to max_heap (negate for max heap behavior)
        heapq.heappush(self.max_heap, -num)

        # Balance: ensure max of left <= min of right
        if self.max_heap and self.min_heap and (-self.max_heap[0] > self.min_heap[0]):
            val = -heapq.heappop(self.max_heap)
            heapq.heappush(self.min_heap, val)

        # Balance sizes: max_heap can have at most 1 more element than min_heap
        if len(self.max_heap) > len(self.min_heap) + 1:
            val = -heapq.heappop(self.max_heap)
            heapq.heappush(self.min_heap, val)
        elif len(self.min_heap) > len(self.max_heap):
            val = heapq.heappop(self.min_heap)
            heapq.heappush(self.max_heap, -val)

    def findMedian(self) -> float:
        """
        Return the median of all numbers added so far.

        Returns
        -------
        float
            The median value of the current data stream.

        Notes
        -----
        If the total number of elements is odd, return the middle element.
        If it is even, return the average of the two middle elements.
        """
        if len(self.max_heap) > len(self.min_heap):
            return -self.max_heap[0]
        else:
            return (-self.max_heap[0] + self.min_heap[0]) / 2.0
