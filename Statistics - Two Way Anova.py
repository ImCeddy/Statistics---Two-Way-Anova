from tracemalloc import Statistic
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

df = pd.DataFrame({'Location': np.repeat(['LocationA', 'LocationB', 'LocationC'], 10),
                   'Time': np.tile(np.repeat(['8:10-9:00', '9:10-10:00', '10:10-11:00'], 1), 10),
                   'Upload': [6.84500, 9.18333, 10.45667, 6.75500, 8.19833, 10.41000, 5.07644, 6.88667, 10.18500, 5.44333,
                                5.82167, 9.98333, 6.59167, 6.98833, 10.14667, 6.63667, 5.17167, 9.79000, 5.16500, 6.69500,
                              9.52500, 6.06667, 6.44167, 10.78167, 5.46833, 5.79500, 10.62333, 7.52667, 5.56500, 10.3667]})

model = ols('Upload ~ C(Location) + C(Time) + C(Location):C(Time)', data=df).fit()
anova_Result = sm.stats.anova_lm(model, typ=2)
print("\nDescriptive Statistic")
print(df.describe())
print(df)
print(anova_Result)