import json
from typing import List

from mrjob.job import MRJob
from mrjob.step import MRStep

class MRCountConsecutiveDecreases(MRJob):
    def configure_args(self):
        super().configure_args()
        self.add_passthru_arg("--k", type=int, required=True)

    def steps(self):
        return [
            MRStep(
                mapper=self.mapper_chunk_summary,
                reducer=self.reducer_total_count,
            )
        ]

    def mapper_chunk_summary(self, _, line):
        """
        Map phase: analyze each chunk and compute summary statistics.

        For each chunk, we:
        1. Count consecutive k-decreases fully within the chunk
        2. Store boundary information to detect cross-chunk sequences
        """
        # Parse input
        record = json.loads(line)
        chunk_id = record["chunk_id"]
        prices = record["prices"]
        k = self.options.k

        # Count consecutive k-decreases within this chunk
        count = 0
        for i in range(len(prices) - k + 1):
            # Check if prices[i] > prices[i+1] > ... > prices[i+k-1]
            is_decreasing = all(prices[j] > prices[j + 1] for j in range(i, i + k - 1))
            if is_decreasing:
                count += 1

        # Store boundary prices for cross-chunk sequence detection
        # We need up to k-1 prices from each end
        first_prices = prices[:min(len(prices), k - 1)]
        last_prices = prices[-min(len(prices), k - 1):]

        # Emit summary
        summary = {
            "chunk_id": chunk_id,
            "count": count,
            "first_prices": first_prices,
            "last_prices": last_prices,
        }

        # Use None as key so all summaries go to the same reducer
        yield None, summary

    def reducer_total_count(self, _, summaries):
        """
        Reduce phase: merge chunk summaries and count cross-chunk sequences.

        We need to:
        1. Sum up all within-chunk counts
        2. Check boundaries between consecutive chunks for cross-chunk sequences
        """
        # Collect and sort all summaries by chunk_id
        summaries_list = sorted(list(summaries), key=lambda x: x["chunk_id"])

        k = self.options.k

        # Sum up all within-chunk counts
        total_count = sum(s["count"] for s in summaries_list)

        # Check cross-chunk sequences
        for i in range(len(summaries_list) - 1):
            curr_chunk = summaries_list[i]
            next_chunk = summaries_list[i + 1]

            # Combine boundary prices: last k-1 of current + first k-1 of next
            boundary_prices = curr_chunk["last_prices"] + next_chunk["first_prices"]

            # Count k-decreases in the boundary region
            # We only count sequences that actually cross the boundary
            # (not already counted within chunks)
            for j in range(len(boundary_prices) - k + 1):
                # Check if this sequence crosses the chunk boundary
                # It crosses if it includes prices from both chunks
                sequence_end = j + k
                crosses_boundary = (j < len(curr_chunk["last_prices"]) and
                                  sequence_end > len(curr_chunk["last_prices"]))

                if crosses_boundary:
                    # Check if it's a decreasing sequence
                    is_decreasing = all(boundary_prices[m] > boundary_prices[m + 1]
                                      for m in range(j, j + k - 1))
                    if is_decreasing:
                        total_count += 1

        yield "total_count", total_count


if __name__ == "__main__":
    MRCountConsecutiveDecreases.run()
