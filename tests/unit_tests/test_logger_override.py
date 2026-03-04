from __future__ import annotations

from opto import trace
from opto.optimizers.optimizer import Optimizer
from opto.trainer.algorithms.algorithm import Trainer
from opto.trainer.guide import Guide
from opto.trainer.loggers import (
    BaseLogger,
    ConsoleLogger,
    DefaultLogger,
    NullLogger,
    list_logger_names,
)
import importlib

train_mod = importlib.import_module("opto.trainer.train")


class DummyOptimizer(Optimizer):
    def _step(self, *args, **kwargs):
        return {}


class DummyGuide(Guide):
    def get_feedback(self, query, response, reference=None, **kwargs):
        return 1.0, "ok"


class NoLoggerTrainer(Trainer):
    """Trainer intentionally lacking `logger` kwarg in __init__."""

    def __init__(self, agent, optimizer):
        super().__init__(agent)
        self.optimizer = optimizer

    def train(self, guide, train_dataset, **kwargs):
        return {
            "ok": True,
            "logger_type": type(self.logger).__name__,
            "dataset_size": len(train_dataset["inputs"]),
        }


def test_list_logger_names_contains_none_and_console():
    names = list_logger_names(include_none=True)
    assert "none" in names
    assert "ConsoleLogger" in names
    assert "NullLogger" not in names


def test_load_logger_supports_none_aliases():
    assert isinstance(train_mod.load_logger("none"), NullLogger)
    assert isinstance(train_mod.load_logger("null"), NullLogger)
    assert isinstance(train_mod.load_logger("off"), NullLogger)
    assert isinstance(train_mod.load_logger("disabled"), NullLogger)
    assert isinstance(train_mod.load_logger(None), DefaultLogger)


def test_load_logger_unknown_falls_back_to_default(capsys):
    logger = train_mod.load_logger("NotARealLogger")
    out = capsys.readouterr().out
    assert isinstance(logger, DefaultLogger)
    assert "Unknown logger" in out


def test_load_logger_accepts_instances_and_classes():
    assert isinstance(train_mod.load_logger(ConsoleLogger()), ConsoleLogger)
    assert isinstance(train_mod.load_logger(ConsoleLogger), ConsoleLogger)


def test_train_retries_without_logger_kwarg(monkeypatch, capsys):
    def _mock_load_optimizer(optimizer, model, **kwargs):
        return DummyOptimizer(model.parameters())

    def _mock_load_guide(guide, **kwargs):
        return DummyGuide()

    monkeypatch.setattr(train_mod, "load_optimizer", _mock_load_optimizer)
    monkeypatch.setattr(train_mod, "load_guide", _mock_load_guide)

    param = trace.node(0, trainable=True)
    train_dataset = {"inputs": [1], "infos": [1]}

    result = train_mod.train(
        model=param,
        train_dataset=train_dataset,
        algorithm=NoLoggerTrainer,
        optimizer="unused",
        guide="unused",
        logger="none",
    )

    out = capsys.readouterr().out
    assert "does not accept logger" in out
    assert result["ok"] is True
    assert result["dataset_size"] == 1
    assert isinstance(train_mod.load_logger("none"), BaseLogger)
