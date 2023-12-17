import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules


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

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

    # Display the top 10 association rules based on lift
    top_rules = rules.nlargest(10, 'lift')[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    print(top_rules)


def Apriori(min_support):
    df = pd.read_csv("./Data/Q4/retail.csv")
    basket = df.groupby(['Transaction_id', 'Product_id'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('Transaction_id')

    basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0) # Convert quantities to binary values (1 if the item was bought, 0 otherwise)

    # Generate frequent itemsets using Apriori algorithm
    frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
    frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)
    print(frequent_itemsets.head(10))

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

    # Display the top 10 association rules based on lift
    top_rules = rules.nlargest(10, 'lift')[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    print(top_rules)


if __name__ == "__main__":
    FP_growth(min_support=0.01)
    print("==============================================================================")
