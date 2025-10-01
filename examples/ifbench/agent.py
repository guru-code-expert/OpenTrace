import datasets
import numpy as np
from opto import trace
from opto.utils.llm import LLM
from opto.features.predefined_agents import BasicLearner
from opto.optimizers import OptoPrimeV2 as OptoPrime
from opto.features.priority_search import PrioritySearch as SearchAlgorithm
from opto.trainer.loggers import TensorboardLogger
from opto.trainer.guide import LLMJudge
from opto.trace.operators import call_llm

"""
The IFBench test set consists of 58 new and out-of-distribution output constraints and
instructions to test system’s ability to generalize to new task constraints. Pyatkin et al. (2025b) also release IFTrain and
IF-RLVR Train data (Pyatkin et al., 2025a) which are used for training. We split the IF-RLVR Train into our train/val sets, and IFBench as our test set.
"""

import datasets

base_prompt = "Respond to the query"
# response_agent = BasicLearner(system_prompt=base_prompt, llm=LLM())

ensure_correctness_prompt = "Ensure the response is correct and adheres to the given constraints. Your response will be used as the final response."
# correctness_agent = BasicLearner(system_prompt=base_prompt, llm=LLM())

@trace.bundle()
def call_llm(llm, system_prompt: str, query: str) -> str:
    """Call LLM with system and user prompts.

    Args:
        llm: The LLM instance to use
        system_prompt: The system prompt for the LLM
        query: The input message to format into the template

    Returns:
        The LLM response content

    Raises:
        ValueError: If user_prompt_template doesn't contain {message}
    """
    response = llm(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content

class IFBenchLearner:
    def __init__(
            self,
            system_prompt1: str = "Respond to the query",
            system_prompt2: str = ensure_correctness_prompt,
            llm: LLM = None
    ):
        """Initialize the learner.

        Args:
            system_prompt: The system prompt to guide LLM behavior
            user_prompt_template: Template for formatting user messages (must contain {message})
            llm: LLM instance to use (defaults to LLM())
        """
        self.system_prompt1 = trace.node(system_prompt1, trainable=True)
        self.system_prompt2 = trace.node(system_prompt2, trainable=True)
        self.llm = llm or LLM()

    def forward(self, message: str) -> str:
        response = call_llm(self.llm, self.system_prompt1, f"{message}")
        final_response = call_llm(self.llm, self.system_prompt2, response)
        return final_response


def main():
    # set seed
    seed = 42
    num_epochs = 1
    batch_size = 3  # number of queries to sample from the training data
    sub_batch_size = 2  # number of queries each optimizer sees
    num_proposals = 3  # number of proposals to generate for each query
    num_candidates = 2  # number of candidates for exploration
    score_range = (0, 1)  # range of the score for the guide
    eval_frequency = -1
    num_eval_samples = 2
    score_function = 'mean'

    num_threads = 10
    datasize = 5
    verbose = True

    np.random.seed(seed)

    # In this example, we use the BBEH dataset
    train_dataset = datasets.load_dataset('allenai/IF_multi_constraints_upto5')['train'][:datasize]
    train_dataset = dict(inputs=train_dataset['input'], infos=train_dataset['target'])

    agent = IFBenchLearner(llm=LLM())
    guide = LLMJudge(llm=LLM())
    optimizer = OptoPrime(agent.parameters(), llm=LLM())
    logger = TensorboardLogger(verbose=verbose)

    alg = SearchAlgorithm(
        agent=agent,
        optimizer=optimizer,
        logger=logger
    )

if __name__ == "__main__":
    main()