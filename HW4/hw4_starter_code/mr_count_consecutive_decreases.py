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
        raise NotImplementedError

    def reducer_total_count(self, _, summaries):
        raise NotImplementedError


if __name__ == "__main__":
    MRCountConsecutiveDecreases.run()
