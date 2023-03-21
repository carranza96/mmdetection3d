import pandas as pd
res = \
{'Vehicle/L1 mAP': 0.664376, 'Vehicle/L1 mAPH': 0.658278, 'Vehicle/L2 mAP': 0.584505, 'Vehicle/L2 mAPH': 0.579036, 'Pedestrian/L1 mAP': 0.655987, 'Pedestrian/L1 mAPH': 0.465067, 'Pedestrian/L2 mAP': 0.577341, 'Pedestrian/L2 mAPH': 0.408068, 'Sign/L1 mAP': 0.0, 'Sign/L1 mAPH': 0.0, 'Sign/L2 mAP': 0.0, 'Sign/L2 mAPH': 0.0, 'Cyclist/L1 mAP': 0.527875, 'Cyclist/L1 mAPH': 0.486404, 'Cyclist/L2 mAP': 0.507904, 'Cyclist/L2 mAPH': 0.467992, 'Overall/L1 mAP': 0.6160793333333333, 'Overall/L1 mAPH': 0.536583, 'Overall/L2 mAP': 0.5565833333333333, 'Overall/L2 mAPH': 0.48503199999999996}

df = (pd.DataFrame(res,index=[0])*100).round(decimals=2)
df = df.drop([x for x in df.columns if 'Sign' in x],axis=1)
df.to_csv("results.csv",index=False)