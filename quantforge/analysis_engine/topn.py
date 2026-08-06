from quantforge.analysis.topn_engine import TopNEngine


def topn(predictions):

    print("=" * 80)
    print("TOP-N")
    print("=" * 80)

    return TopNEngine(predictions)