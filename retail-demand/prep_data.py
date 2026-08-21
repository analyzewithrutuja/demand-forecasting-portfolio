import pandas as pd

xls = pd.ExcelFile("online_retail_II.xlsx")
dfs = [pd.read_excel("online_retail_II.xlsx", sheet_name=s) for s in xls.sheet_names]
df = pd.concat(dfs, ignore_index=True)
df["Invoice"] = df["Invoice"].astype(str)
print(len(df), df["InvoiceDate"].min(), df["InvoiceDate"].max())
df.to_csv("online_retail_combined.csv", index=False)
print("saved csv")
