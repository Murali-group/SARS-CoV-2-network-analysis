
import pandas as pd


def parse_drugbank_csv(csv, **kwargs):
    df = pd.read_csv(csv, sep=',', index_col=0)

    col_to_split = 'Drug IDs'
    new_col = 'Drug ID'
    df[col_to_split] = df[col_to_split].apply(lambda x: x.split('; '))
    orig_cols = [c for c in df.columns if c != col_to_split]
    # we want to put each Drug ID on its own row, with the rest of the row copied. 
    # this gets the job done
    df2 = pd.concat([df[col_to_split].apply(pd.Series), df], axis=1) \
        .drop(col_to_split, axis=1) \
        .melt(id_vars=orig_cols, value_name=new_col) \
        .drop("variable", axis=1) \
        .dropna(subset=[new_col])

    #print(df2.head())
    return df2
