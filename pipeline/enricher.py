from __future__ import annotations

from pipeline.classifiers.question_type import QuestionTypeClassifier
from pipeline.classifiers.difficulty import DifficultyClassifier
from pipeline.classifiers.topic_classifier import TopicClassifier
from pipeline.classifiers.bloom_classifier import BloomClassifier
from pipeline.classifiers.answer_type_classifier import AnswerTypeClassifier


class QuestionEnricher:

    def __init__(self):

        self.question_classifier = QuestionTypeClassifier()
        self.difficulty_classifier = DifficultyClassifier()
        self.topic_classifier = TopicClassifier()
        self.bloom_classifier = BloomClassifier()
        self.answer_classifier = AnswerTypeClassifier()

    def process(self, questions):

        for q in questions:

            # -------------------------
            # Question Type
            # -------------------------

            q.question_type = self.question_classifier.classify(q)

            # -------------------------
            # Difficulty
            # -------------------------

            q.difficulty = self.difficulty_classifier.classify(q)

            # -------------------------
            # Topic
            # -------------------------

            q.topic = self.topic_classifier.classify(q)

            # -------------------------
            # Bloom
            # -------------------------

            q.bloom_level = self.bloom_classifier.classify(q)

            # -------------------------
            # Expected Answer Type
            # -------------------------

            q.answer_type = self.answer_classifier.classify(q)

        return questions
