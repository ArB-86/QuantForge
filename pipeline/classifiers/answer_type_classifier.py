from __future__ import annotations


class AnswerTypeClassifier:
    """
    Expected answer format.

    Returns one of:

        option
        boolean
        number
        equation
        proof
        graph
        construction
        explanation
    """

    def classify(self, question):

        text = question.question.lower()

        # MCQ

        if getattr(question, "options", None):
            return "option"

        # True / False

        if (
            "true or false" in text
            or "is it true" in text
            or "justify whether" in text
        ):
            return "boolean"

        # Proof

        if (
            "prove that" in text
            or "show that" in text
            or "verify that" in text
        ):
            return "proof"

        # Graph

        if (
            "draw the graph" in text
            or "graphically" in text
            or "plot" in text
        ):
            return "graph"

        # Construction

        if (
            "construct" in text
            or "construction" in text
        ):
            return "construction"

        # Equation

        if (
            "solve the equation" in text
            or "roots of" in text
            or "find x" in text
            or "find the value of x" in text
        ):
            return "equation"

        # Numerical

        if (
            "find" in text
            or "calculate" in text
            or "evaluate" in text
            or "determine" in text
            or "compute" in text
        ):
            return "number"

        return "explanation"
