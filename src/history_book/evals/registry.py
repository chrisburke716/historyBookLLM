"""
Registry system for discovering and managing evaluators.
"""

from history_book.evals.base import BaseEvaluator, FunctionEvaluator

# Global registry of evaluators
_EVALUATOR_REGISTRY: dict[str, type[BaseEvaluator]] = {}


def register_evaluator(evaluator_class: type[BaseEvaluator]) -> type[BaseEvaluator]:
    """
    Decorator to register an evaluator class.

    Usage:
        @register_evaluator
        class MyEvaluator(BaseEvaluator):
            # implementation
    """
    # Create a temporary instance to get the name
    temp_instance = evaluator_class()
    name = temp_instance.name

    if name in _EVALUATOR_REGISTRY:
        raise ValueError(f"Evaluator with name '{name}' is already registered")

    _EVALUATOR_REGISTRY[name] = evaluator_class
    return evaluator_class


def get_evaluator(name: str) -> type[BaseEvaluator]:
    """
    Get an evaluator class by name.

    Args:
        name: Name of the evaluator

    Returns:
        Evaluator class

    Raises:
        KeyError: If evaluator with given name is not registered
    """
    if name not in _EVALUATOR_REGISTRY:
        available = list(_EVALUATOR_REGISTRY.keys())
        raise KeyError(f"Evaluator '{name}' not found. Available: {available}")

    return _EVALUATOR_REGISTRY[name]


def list_evaluators() -> list[str]:
    """
    List all registered evaluator names.

    Returns:
        List of evaluator names
    """
    return list(_EVALUATOR_REGISTRY.keys())


def create_evaluator(name: str, **kwargs) -> BaseEvaluator:
    """
    Create an evaluator instance by name.

    Args:
        name: Name of the evaluator
        **kwargs: Arguments to pass to evaluator constructor

    Returns:
        Evaluator instance
    """
    evaluator_class = get_evaluator(name)
    return evaluator_class(**kwargs)


def get_all_evaluators(**kwargs) -> list[BaseEvaluator]:
    """
    Create instances of all registered evaluators.

    Args:
        **kwargs: Arguments to pass to all evaluator constructors

    Returns:
        List of evaluator instances
    """
    return [create_evaluator(name, **kwargs) for name in list_evaluators()]


def get_prompt_evaluators(**kwargs) -> list[BaseEvaluator]:
    """
    Create instances of all prompt-based evaluators (Criteria and LabeledCriteria).

    Args:
        **kwargs: Arguments to pass to evaluator constructors

    Returns:
        List of prompt-based evaluator instances
    """
    prompt_evaluators = []
    for name in list_evaluators():
        evaluator = create_evaluator(name, **kwargs)
        if not isinstance(evaluator, FunctionEvaluator):
            prompt_evaluators.append(evaluator)
    return prompt_evaluators


def get_function_evaluators(**kwargs) -> list[BaseEvaluator]:
    """
    Create instances of all function-based evaluators.

    Args:
        **kwargs: Arguments to pass to evaluator constructors

    Returns:
        List of function-based evaluator instances
    """
    function_evaluators = []
    for name in list_evaluators():
        evaluator = create_evaluator(name, **kwargs)
        if isinstance(evaluator, FunctionEvaluator):
            function_evaluators.append(evaluator)
    return function_evaluators
