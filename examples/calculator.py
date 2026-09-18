"""Demonstration module for unit test generation and pair-programming evaluation."""


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtract the second number from the first."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide the first number by the second.

    Raises:
        ValueError: If denominator (b) is zero.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b
