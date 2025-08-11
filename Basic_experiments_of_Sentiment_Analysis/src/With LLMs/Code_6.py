import pandas as pd
import os
import dashscope
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

api_key=os.getenv("DASHSCOPE_API_KEY")

if api_key is None:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY。")

dashscope.api_key = api_key

data=pd.read_csv("D:\Code\Sentiment Analysis\Data\IMDB Dataset.csv")
data['sentiment'] = data['sentiment'].apply(lambda x: 'Positive' if x == 'positive' else 'Negative')
data_test=data.sample(n=10000,random_state=42)
print(f"测试集大小: {len(data_test)}")

def get_prediction_from_api(text):
    messages=[
        {'role':'system','content':'你是一名情感分析助手，请对用户给出的电影评论进行情感分析。'},
        {'role':'user','content': f'对以下电影评论进行情感分析，并只输出"Positive"或"Negative":\n\n评论:{text}'}
    ]
    try:
        response = dashscope.Generation.call(
            model="qwen-plus",
            messages=messages,
            result_format='message',
            temperature=0.1
        )
        if response.status_code == 200:
            content = response.output.choices[0].message.content
            if "Positive" in content:
                return "Positive"
            elif "Negative" in content:
                return "Negative"
            else:
                return "Unknown"
        else:
            print(f"API 调用失败！状态码: {response.status_code}")
            print(f"错误码: {response.code}")
            print(f"错误信息: {response.message}")
            return "API_Error"
    except Exception as e:
        print(f"API 调用出错: {e}")
        return "API_Error"
    
    
data_test['predicted_sentiment']=None
for index,row in tqdm(data_test.iterrows(),total=len(data_test)):
    review=row['review']
    predicted=get_prediction_from_api(review)
    data_test.loc[index,'predicted_sentiment']=predicted
    

data_filtered = data_test[~data_test['predicted_sentiment'].isin(['Unknown', 'API_Error'])]
accuracy = accuracy_score(data_filtered['sentiment'], data_filtered['predicted_sentiment'])

print("\n--- 评估结果 ---")
print(f"总测试样本数: {len(data_test)}")
print(f"有效预测样本数: {len(data_filtered)}")
print(f"准确率: {accuracy:.4f}")
print("\n分类报告:")
print(classification_report(data_filtered['sentiment'], data_filtered['predicted_sentiment']))