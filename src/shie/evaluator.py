
class SHIEEvaluator:

    @staticmethod
    def levenshtein_distance(reference, prediction):
        """Calculate minimum edit distance."""

        reference_length = len(reference)
        prediction_length = len(prediction)

        dp = [
            [0] * (prediction_length + 1)
            for _ in range(reference_length + 1)
        ]

        for i in range(reference_length + 1):
            dp[i][0] = i

        for j in range(prediction_length + 1):
            dp[0][j] = j

        for i in range(1, reference_length + 1):
            for j in range(1, prediction_length + 1):

                cost = (
                    0 if reference[i - 1] == prediction[j - 1]
                    else 1
                )

                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost
                )

        return dp[reference_length][prediction_length]

    @staticmethod
    def calculate_cer(reference, prediction):
        """Calculate Character Error Rate."""

        if not isinstance(reference, str):
            raise TypeError("Reference must be a string.")

        if not isinstance(prediction, str):
            raise TypeError("Prediction must be a string.")

        if not reference:
            return 0.0 if not prediction else 1.0

        distance = SHIEEvaluator.levenshtein_distance(
            reference,
            prediction
        )

        return distance / len(reference)

    @staticmethod
    def calculate_wer(reference, prediction):
        """Calculate Word Error Rate."""

        if not isinstance(reference, str):
            raise TypeError("Reference must be a string.")

        if not isinstance(prediction, str):
            raise TypeError("Prediction must be a string.")

        reference_words = reference.split()
        prediction_words = prediction.split()

        if not reference_words:
            return 0.0 if not prediction_words else 1.0

        distance = SHIEEvaluator.levenshtein_distance(
            reference_words,
            prediction_words
        )

        return distance / len(reference_words)

    @staticmethod
    def edit_statistics(reference, prediction):
        """
        Calculate substitutions, deletions and insertions
        using dynamic programming and backtracking.

        Both arguments must be sequences.
        """

        m = len(reference)
        n = len(prediction)

        dp = [
            [0] * (n + 1)
            for _ in range(m + 1)
        ]

        for i in range(m + 1):
            dp[i][0] = i

        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):

                cost = 0 if reference[i - 1] == prediction[j - 1] else 1

                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost
                )

        substitutions = 0
        deletions = 0
        insertions = 0

        i = m
        j = n

        # Backtrack from the final cell.
        while i > 0 or j > 0:

            # Matching characters or words.
            if (
                i > 0
                and j > 0
                and reference[i - 1] == prediction[j - 1]
                and dp[i][j] == dp[i - 1][j - 1]
            ):
                i -= 1
                j -= 1

            # Substitution.
            elif (
                i > 0
                and j > 0
                and dp[i][j] == dp[i - 1][j - 1] + 1
            ):
                substitutions += 1
                i -= 1
                j -= 1

            # Deletion.
            elif (
                i > 0
                and dp[i][j] == dp[i - 1][j] + 1
            ):
                deletions += 1
                i -= 1

            # Insertion.
            elif (
                j > 0
                and dp[i][j] == dp[i][j - 1] + 1
            ):
                insertions += 1
                j -= 1

            else:
                raise RuntimeError("Invalid edit-distance backtracking.")

        return {
            "substitutions": substitutions,
            "deletions": deletions,
            "insertions": insertions,
            "total_errors": substitutions + deletions + insertions
        }

    @staticmethod
    def _evaluate_dataset(references, predictions, word_level=False):
        """
        Aggregate evaluation across multiple samples.

        Returns aggregate error rate and edit statistics.
        """

        if not isinstance(references, (list, tuple)):
            raise TypeError("References must be a list or tuple.")

        if not isinstance(predictions, (list, tuple)):
            raise TypeError("Predictions must be a list or tuple.")

        if len(references) != len(predictions):
            raise ValueError(
                "Reference and prediction counts must match."
            )

        if not all(isinstance(x, str) for x in references):
            raise TypeError("Every reference must be a string.")

        if not all(isinstance(x, str) for x in predictions):
            raise TypeError("Every prediction must be a string.")

        totals = {
            "samples": len(references),
            "reference_units": 0,
            "substitutions": 0,
            "deletions": 0,
            "insertions": 0,
            "total_errors": 0,
            "empty_reference_samples": 0
        }

        for reference, prediction in zip(references, predictions):

            if word_level:
                reference_units = reference.split()
                prediction_units = prediction.split()
            else:
                reference_units = list(reference)
                prediction_units = list(prediction)

            if not reference_units:
                totals["empty_reference_samples"] += 1

            statistics = SHIEEvaluator.edit_statistics(
                reference_units,
                prediction_units
            )

            totals["reference_units"] += len(reference_units)

            for key in (
                "substitutions",
                "deletions",
                "insertions",
                "total_errors"
            ):
                totals[key] += statistics[key]

        if totals["reference_units"] == 0:
            totals["error_rate"] = None
        else:
            totals["error_rate"] = (
                totals["total_errors"] / totals["reference_units"]
            )

        return totals

    @staticmethod
    def evaluate_cer_dataset(references, predictions):
        """Evaluate aggregate Character Error Rate."""

        return SHIEEvaluator._evaluate_dataset(
            references,
            predictions,
            word_level=False
        )

    @staticmethod
    def evaluate_wer_dataset(references, predictions):
        """Evaluate aggregate Word Error Rate."""

        return SHIEEvaluator._evaluate_dataset(
            references,
            predictions,
            word_level=True
        )
