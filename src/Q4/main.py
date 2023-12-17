import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth


def FP_growth(min_support):
    df = pd.read_csv("./Data/Q4/retail.csv")
    Transaction_df = df.groupby('Transaction_id')['Product_id'].apply(list).reset_index(name='items')
    data = Transaction_df["items"].tolist()

    te = TransactionEncoder()
    te_ary = te.fit(data).transform(data)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
    frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)

    print(frequent_itemsets.head(10))


if __name__ == "__main__":
    FP_growth(min_support=0.1)