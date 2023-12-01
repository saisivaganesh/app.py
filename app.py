import streamlit as st
import pandas as pd
from pandas_datareader import data as pdr
import datetime
import yfinance as yf
from prophet import Prophet
import matplotlib.pyplot as plt

#BTCのチャートを取得
tickers = ['BTC-JPY']
#取得開始日を入力
start = "2020-03-01"
#取得終了日を入力
end = datetime.date.today()
#Yahoofinanceから取得するように設定
yf.pdr_override()
#データの取得を実行
crypto_data = pdr.get_data_yahoo(tickers, start, end)

#予想したい日付を入力
periods=90
# インスタンス化
model = Prophet()
# モデルフィット
df = pd.DataFrame({"ds":crypto_data['Adj Close'].index, "y":crypto_data['Adj Close']}).reset_index(drop=True)
model.fit(df)
# 未来予測用のデータフレーム
future = model.make_future_dataframe(periods=periods)
# 時系列を予測
forecast = model.predict(future)
#表示
fig = model.plot(forecast)
ax = fig.gca()
ax.set_title('ビットコイン3ヵ月予測チャート',fontname="UD Digi Kyokasho N-B", fontsize=21)
arrowprops=dict(arrowstyle='->')

st.pyplot(fig)
# グラフの表示
fig.show()
