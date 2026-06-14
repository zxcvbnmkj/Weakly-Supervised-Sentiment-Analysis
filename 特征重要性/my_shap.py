import pandas as pd
import shap
import sklearn

df=pd.read_csv("D:\Pythonnnn\弱监督的农业社会化销售服务评价\特征重要性\data.csv")
group=['品质','味道','价格','分量','外观','物流','客服']
X=df[group]
y=df["用户评分"]

X100 = shap.utils.sample(X, 100) # 100 instances for use as the background distribution



#使用线性模型，输出每个特征的系数。它表示系数每改变1，y值会发生的变化
# model = sklearn.linear_model.LinearRegression()
# model.fit(X, y)
# print("Model coefficients:\n")
# for i in range(X.shape[1]):
#     print(X.columns[i], "=", model.coef_[i].round(5))
"""
当标签是0 1 2时
品质 = 0.05672
味道 = 0.13947
价格 = -0.28461
分量 = 0.00707
外观 = -0.0359
物流 = -0.02369
客服 = -0.02126

当标签是-1 0 1时
品质 = 0.34314
味道 = 0.51085
价格 = 0.04935
分量 = 0.12247
外观 = 0.11186
物流 = 0.13139
客服 = 0.22635

在线性模型中，每个特征都被单独处理，并且把它们的影响简单地相加。我们希望在放松模型对于直线关系的要求的同时，保持这种可加性质。这就引出了广义可加模型（GAMs）
"""

import interpret.glassbox
model_ebm = interpret.glassbox.ExplainableBoostingRegressor(interactions=0)
model_ebm.fit(X, y)

# explain the GAM model with SHAP
explainer_ebm = shap.Explainer(model_ebm.predict, X100)
shap_values_ebm = explainer_ebm(X)

sample_ind=20
# make a standard partial dependence plot with a single SHAP value overlaid
fig,ax = shap.partial_dependence_plot(
    "品质", model_ebm.predict, X100, model_expected_value=True,
    feature_expected_value=True, show=False, ice=False,
    shap_values=shap_values_ebm[sample_ind:sample_ind+1,:]
)
shap.plots.waterfall(shap_values_ebm[sample_ind])
shap.plots.beeswarm(shap_values_ebm)
# shap.plots.bar(shap_values)