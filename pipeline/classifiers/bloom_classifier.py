from __future__ import annotations


class BloomClassifier:
    """
    Bloom Taxonomy classifier.

    Levels:
        remember
        understand
        apply
        analyze
        evaluate
        create
    """

    RULES = {
        "remember": [
            "define",
            "state",
            "list",
            "identify",
            "name",
            "write",
        ],

        "understand": [
            "explain",
            "describe",
            "justify",
            "interpret",
            "why",
            "reason",
        ],

        "apply": [
            "find",
            "calculate",
            "solve",
            "evaluate",
            "determine",
            "compute",
            "obtain",
        ],

        "analyze": [
            "prove",
            "show that",
            "verify",
            "graphically",
            "compare",
            "distinguish",
            "analyse",
            "analyze",
        ],

        "evaluate": [
            "is it true",
            "true or false",
            "comment",
            "justify whether",
            "examine",
            "check",
        ],

        "create": [
            "construct",
            "draw",
            "design",
            "sketch",
            "plot",
        ],
    }

    PRIORITY = [
        "create",
        "evaluate",
        "analyze",
        "apply",
        "understand",
        "remember",
    ]

    def classify(self, question):

        text = question.question.lower()

        scores = {}

        for level, words in self.RULES.items():

            score = 0

            for w in words:

                if w in text:
                    score += 1

            scores[level] = score

        best = max(scores.values())

        if best == 0:
            return "apply"

        for level in self.PRIORITY:

            if scores[level] == best:
                return level

        return "apply"
