import datasets
import numpy as np
from opto import trace
from opto.utils.llm import LLM
from opto.features.predefined_agents import BasicLearner
from opto.optimizers import OptoPrimeV2 as OptoPrime
from opto.features.priority_search import PrioritySearch as SearchAlgorithm
from opto.trainer.loggers import TensorboardLogger
from opto.trainer.guide import LLMJudge

"""
The IFBench test set consists of 58 new and out-of-distribution output constraints and
instructions to test system’s ability to generalize to new task constraints. Pyatkin et al. (2025b) also release IFTrain and
IF-RLVR Train data (Pyatkin et al., 2025a) which are used for training. We split the IF-RLVR Train into our train/val sets, and IFBench as our test set.
"""

import datasets

