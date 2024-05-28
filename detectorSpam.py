'''
Detector de spam

@author: Lucivaldo Barbosa 

'''

#importacao do dataset

from ucimlrepo import fetch_ucirepo 

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, classification_report

# fetch dataset 
spambase = fetch_ucirepo(id=94) 
  
# data (as pandas dataframes) 
X = spambase.data.features 
y = spambase.data.targets

X_spam = X.values

y_spam = y.values.reshape(4601,)

X_spam_treinamento, X_spam_teste, y_spam_treinamento, y_spam_teste = train_test_split(X_spam, y_spam, train_size= 0.15, random_state=0)
  

rede_neural = MLPClassifier(activation='relu', hidden_layer_sizes=(10,10), max_iter=1500, solver='adam', verbose=True, tol=0.00001)

rede_neural.fit(X_spam_treinamento, y_spam_treinamento)

previsoes = rede_neural.predict(X_spam_teste)

precisao = accuracy_score(y_spam_teste, previsoes)

print(precisao)

print(classification_report(y_spam_teste, previsoes))