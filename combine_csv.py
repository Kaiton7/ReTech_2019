import pandas as pd

data1 = pd.read_csv("predic_fill_lgbm_sub.csv")
data2 = pd.read_csv("lgbm_1.csv")
data3 = pd.read_csv("lgbm.csv")
data4 = pd.read_csv("all_lgbm_sub_first.csv")


vals = (data1.SalePrice + data2.SalePrice + data3.SalePrice+ data4.SalePrice)/4

print(vals)

res = pd.read_csv("sample_submission.csv")
res["SalePrice"] = vals

res.to_csv("comb_4.csv", index=False)
