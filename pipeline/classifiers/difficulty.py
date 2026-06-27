from __future__ import annotations

import re


class DifficultyClassifier:
    """
    Rule-based difficulty classifier.

    Returns:
        easy
        medium
        hard
    """

    PROOF = (
        "prove",
        "show that",
        "justify",
        "verify",
        "establish",
        "derive",
    )

    GRAPH = (
        "graph",
        "graphically",
        "plot",
        "draw the graph",
    )

    CONSTRUCTION = (
        "construct",
        "construction",
    )

    MULTISTEP = (
        "hence",
        "therefore",
        "also find",
        "find the value of",
        "find the area",
        "find the volume",
        "determine",
    )

    def classify(self, question):

        text = question.question.lower()

        score = 0

        # ------------------------------------
        # Question type
        # ------------------------------------

        qt = getattr(question, "question_type", "")

        if qt == "mcq":
            score += 0

        elif qt == "true_false":
            score += 0

        elif qt == "fill_blank":
            score += 0

        elif qt == "numerical":
            score += 1

        elif qt == "graph":
            score += 2

        elif qt == "construction":
            score += 2

        elif qt == "proof":
            score += 4

        # ------------------------------------
        # Length
        # ------------------------------------

        words = len(text.split())

        if words > 30:
            score += 1

        if words > 60:
            score += 1

        if words > 100:
            score += 1

        # ------------------------------------
        # Keywords
        # ------------------------------------

        if any(k in text for k in self.PROOF):
            score += 3

        if any(k in text for k in self.GRAPH):
            score += 2

        if any(k in text for k in self.CONSTRUCTION):
            score += 2

        if any(k in text for k in self.MULTISTEP):
            score += 1

        # ------------------------------------
        # Mathematical complexity
        # ------------------------------------

        operators = len(re.findall(r"[=+\-×÷*/]", text))

        if operators >= 3:
            score += 1

        if operators >= 8:
            score += 1

        # ------------------------------------
        # Multiple equations
        # ------------------------------------

        equations = text.count("=")

        if equations >= 2:
            score += 1

        # ------------------------------------
        # Figures
        # ------------------------------------

        if "fig." in text or "figure" in text:
            score += 1

        # ------------------------------------
        # Final mapping
        # ------------------------------------

        if score <= 2:
            return "easy"

        if score <= 5:
            return "medium"

        return "hard"
