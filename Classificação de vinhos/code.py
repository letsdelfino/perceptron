#importando bibliotecas

from subprocess import check_output #mostra o arquivo da database seno usado
import pandas as pd #usada para manipulação de arquivos
#import tensorflow as tf
import numpy as np #uado para criar array
import keras
from keras import optimizers
from keras.optimizers import Adam, rmsprop, sgd
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Activation

#mostrando arquivo que está sendo usado
print(check_output(["ls", "../input"]).decode("utf8"))

#incianco leitura do arquivo
dados = pd.read_csv('../input/Wine.csv', header=None)
dados.columns = ['Classe','Álcool','Ácido málico','Cinza','Alcalinidade das cinzas','Magnésio','totalPhenols','Fenóis totais','Fenol não flavonóide' ,'Proantocianimas','Intensidade da cor','Tom','OD280/OD315','prolineProlina']
print(dados)

#eliminando a primeira coluna lida do arquivo
X_entrada = dados.drop(['Classe'], axis=1)
Y_saida = dados.iloc[:,:1] #seleciona linhas e colunas por números 
#https://medium.com/horadecodar/data-science-tips-02-como-usar-loc-e-iloc-no-pandas-fab58e214d87

#convertendo dados para array
#devem ser duas matrizes, uma com as características e outra com as classes. As saídas devem ser passadas para uma matriz onde sua representação será feita com zeros e uns
X = X_entrada.values
Y = Y_saida.values

#fazendo a verificação da conversão
print(type(X), type(Y))

#criando a representação das classes
def classificador(classificacao):
    if classificacao == 1:
        return [1, 0, 0] #equivalente a classe um
    if classificacao == 2:
        return [0, 1, 0] #equivalente a classe dois
    if classificacao == 3:
        return [0, 0, 1] #equivalente a classe três
      
 #salvando novas valores gerados pela função de classificação criada no array Y

Y = np.array([classificador(i[0]) for i in Y])

#verificando tamanho dos conjuntos de treino
print(X.shape)
print(Y.shape)

#criando modelo
model = Sequential([
    Dense(12, input_dim=13, activation='relu'),
    Dense(8, input_dim=13, activation='relu'),
    Dense(3, activation='softmax')
              ])

#compilando modelo

sgd = optimizers.SGD(lr=0.001)
model.compile(
  optimizer='sgd',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

#treinando a rede usando o arquivo todo
model.fit(X,Y,epochs=1000, batch_size=178)

predictions = model.predict(X)
print("predictions",predictions[0])

score = model.evaluate(X,Y,verbose=0)
print(score)
