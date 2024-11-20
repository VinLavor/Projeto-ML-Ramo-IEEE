import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, LSTM, Dropout 


vale = yf.download("VALE3.SA", start = "2024-09-02", end = "2024-11-20")['Adj Close']
vale_train = yf.download("VALE3.SA", start = "2019-01-01" , end = "2024-09-01")['Adj Close']



def plots(df):
    vale_media = df.mean()
    vale_media_movel = df.rolling(window=30).mean()
    vale_desvio = df.std()
    vale_desvio_movel = df.rolling(window=30).std()

    vale_desvio_lista = []
    vale_media_lista = []

    for i in range(0, len(vale)):
        vale_desvio_lista.append(vale_desvio)
        vale_media_lista.append(vale_media)  
    
    plt.plot(vale_media_movel, label='Média Móvel (30 dias)', color='orange')
    plt.plot(vale.index,vale_desvio_lista, label='Desvio Padrão ', color='purple')
    plt.plot(vale.index,vale_media_lista, label='Media ', color='black')
    plt.plot(vale_desvio_movel.index,vale_desvio_movel, label='Desvio Padrão Móvel (30 dias)', color='red')    

train = vale_train.values

def criar_sequencias(dados, n_passos):
    X, y = [], []
    for i in range(n_passos, len(dados)):
        X.append(dados[i-n_passos:i])  
        y.append(dados[i])
    return np.array(X), np.array(y)

x, y = criar_sequencias(train, 40)

x = x.reshape((x.shape[0], x.shape[1], 1))

# criação do modelo 
model = Sequential()
model.add(LSTM(128,return_sequences = True, input_shape = (40,1)))
model.add(LSTM(64,return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mape'])

print(model.summary())

history = model.fit(x, y, batch_size = 32, epochs = 50)

x2, _ = criar_sequencias(vale.values, 40)
x2 = x.reshape((x.shape[0], x.shape[1], 1))

predict = model.predict(x2)
print(predict)

plt.figure(figsize = (12,8))
plt.plot(vale.index[40:], predict, label = "Valores previstos pelo modelo", color = "red")
plt.plot(vale.index[40:], vale.values[40:], label = "Valores reais da ação", color = "black")
plt.title("Predições do modelo de 02/09 até 20/11 ")
plt.xlabel("Data")
plt.ylabel("Preço de fechamento da ação")
plt.legend(fontsize=12)
plt.grid(alpha=0.3)


plt.show()
plt.show()