from pyspark.ml.fpm import FPGrowth
from pyspark.sql import functions as F


def FpTree_Wrapper(df, min_support, min_confidence):
    '''
    FP-Tree Wrapper from Apache FP-Tree algorithm
    '''
    grouped_products = df.groupBy('Transaction_id').agg(F.collect_list('Product_id').alias('items'))
    fpGrowth = FPGrowth(itemsCol="items", minSupport=min_support, minConfidence=min_confidence)
    model = fpGrowth.fit(grouped_products)

    # Display frequent itemsets.
    print("===Frequent Itemsets===")
    model.freqItemsets.show()

    # Display generated association rules.
    print("===Association Rules===")
    model.associationRules.show()

    # transform examines the input items against all the association rules and summarize the
    # consequents as prediction
    print("===Summarize===")
    model.transform(grouped_products).show()

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    data = spark.read.csv("./Data/Q4/retail.csv", header=True)
    data.dropna()
    FpTree_Wrapper(data, min_support=0.4, min_confidence=0.8)