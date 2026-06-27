def calculate_turnover(
    previous,
    current
):
    """
    Portfolio turnover.
    """

    if len(previous) == 0:
        return 1.0

    overlap = len(
        previous &
        current
    )

    return (
        len(current)
        - overlap
    ) / len(current)
