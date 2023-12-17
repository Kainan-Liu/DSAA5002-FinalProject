from pyspark.ml.fpm import PrefixSpan
import pandas as pd


def PrefixSpan_Wrapper(df: pd.DataFrame, min_support, max_length: int):
    prefixSpan = PrefixSpan(minSupport=min_support,
                            maxPatternLength=max_length,
                            maxLocalProjDBSize=32000000)

    # Find frequent sequential patterns.
    prefixSpan.findFrequentSequentialPatterns(df).show()
