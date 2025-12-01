import dspy
import heapq
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))


@dataclass
class HistoryNode:
    """A node representing a conversation history with its associated score."""
    history: dspy.History
    score: float
    depth: int  # Number of questions asked so far
    last_question: str
    last_answer: str

    def __lt__(self, other):
        # For max heap behavior, negate the score (heapq is min heap by default)
        return self.score > other.score

    def __repr__(self):
        return f"HistoryNode(score={self.score}, depth={self.depth}, last_q='{self.last_answer[:30]}...')"


class PriorityQueue:
    """Priority queue for storing conversation histories ranked by score."""

    def __init__(self, max_size: Optional[int] = None):
        self.heap: List[HistoryNode] = []
        self.max_size = max_size

    def pop(self) -> Optional[HistoryNode]:
        """Remove and return the highest-scoring history node."""
        if self.heap:
            return heapq.heappop(self.heap)
        return None

    def peek(self) -> Optional[HistoryNode]:
        """Return the highest-scoring history node without removing it."""
        return self.heap[0] if self.heap else None

    def is_empty(self) -> bool:
        """Check if the priority queue is empty."""
        return len(self.heap) == 0

    def size(self) -> int:
        """Return the number of items in the queue."""
        return len(self.heap)

    def get_all_sorted(self) -> List[HistoryNode]:
        """Return all nodes sorted by score (highest first) without modifying the queue."""
        return sorted(self.heap, reverse=True)

    def clear(self):
        """Remove all items from the queue."""
        self.heap.clear()


class HistoryPriorityQueue(PriorityQueue):

    # We keep the signature specific functions here to change things!
    def push(self, history: dspy.History, score: float, question: str, answer: str, depth: int = 0):
        """Add a new history node to the priority queue."""
        node = HistoryNode(
            history=history,
            score=score,
            depth=depth,
            last_question=question,
            last_answer=answer
        )

        heapq.heappush(self.heap, node)

        # Maintain max size if specified
        if self.max_size and len(self.heap) > self.max_size:
            # Remove the lowest scoring item (at the end after popping highest)
            temp_items = []
            # Keep the best items
            for _ in range(min(self.max_size, len(self.heap))):
                if self.heap:
                    temp_items.append(heapq.heappop(self.heap))

            self.heap = temp_items
            # Re-heapify
            heapq.heapify(self.heap)

class MySignature(dspy.Signature):
    question: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    answer: str = dspy.OutputField()

predict = dspy.Predict(MySignature)
outputs = predict(question="What is the capital of France?")
history = dspy.History(messages=[{"question": "What is the capital of France?", **outputs}])
outputs_with_history = predict(question="Are you sure?", history=history)

"""
Idea 1: Greedy explorer -- always pop off the highest scoring node
Idea 2: Discounted sum of reward -- after each proposal, we update the full PATH that leads to the node (need additional structure)
"""

class GreedyExplorer:
    def __init__(self, exploration_budget: int = 20, max_queue_size: int = 100):
        self.pq = HistoryPriorityQueue(max_size=max_queue_size)
        self.exploration_budget = exploration_budget
        self.explored_nodes = 0
        self.final_result = None
        self.initial_history = dspy.History(messages=[])

if __name__ == '__main__':
    pass