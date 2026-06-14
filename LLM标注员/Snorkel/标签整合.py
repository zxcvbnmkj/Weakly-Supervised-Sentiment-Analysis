import pandas as pd

weidus = ['品质', '味道', '价格', '分量', '外观', '物流', '客服', '粗粒度']

df_model=pd.read_csv("D:\Pythonnnn\弱监督的农业社会化销售服务评价\data\红心柚\弱标签评论集_未修改版.csv")

for weidu in weidus:
    df_ft=pd.read_excel(f"result_{weidu}.xlsx")

    text=[]
    ft_label=[]
    model_label=[]

    # 遍历 df_ft 中的每一行，查找匹配的评论
    for index, row in df_ft.iterrows():
        text_to_match = row['text']
        # 在 df_model 中找到匹配的行
        matched_row = df_model[df_model['评论'] == text_to_match]

        if not matched_row.empty:
            # 获取匹配到的评论内容:Series类型的，(1435, '大果 小果 一起 买 的 ， 建议 大果')
            comment = matched_row['评论']
            comment = comment.iloc[0]
            comment=comment.replace(" ", "")
            # 获取 df_ft 中的 given_label
            given_label = row['given_label']
            # 获取 df_ft 中的 weidu 对应的列:[(1435, 2)]
            #为了取出元组第二个元素，所以加上[1]

            model_weidu_column_value = matched_row[weidu].iloc[0]


            text.append(comment)
            ft_label.append(given_label)
            model_label.append(model_weidu_column_value)

    df_result = pd.DataFrame(data={'评论':text, 'FT':ft_label, 'MODEL':model_label})
    df_result.to_csv(f"{weidu}维度.csv",index=False)
