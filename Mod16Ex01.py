import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport

import seaborn as sns

from matplotlib import pyplot
import matplotlib.pyplot as plt
#%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

from scipy.stats import ks_2samp
import statsmodels.formula.api as smf
import statsmodels.api as sm

from datetime import datetime

import os
import sys

renda = pd.read_csv('./previsao_de_renda.csv')
renda.head(1)

prof = ProfileReport(renda, explorative=True, minimal=True)
prof.to_file('./output/analise_renda.html')
prof

renda.drop(['Unnamed: 0', 'id_cliente'], axis=1)
renda['posse_de_veiculo'] = renda['posse_de_veiculo'].map({True: 1,False: 0})
renda['posse_de_imovel'] = renda['posse_de_imovel'].map({True: 1,False: 0})
renda['data_ref'] = pd.to_datetime(renda['data_ref'])
renda['data_ref'] = renda['data_ref'].dt.strftime('%m-%Y')
renda.head()

os.makedirs('./output/figs/', exist_ok=True)

plt.figure(figsize=(25, 18))
sns.countplot(x= renda['data_ref'],  hue = renda['sexo'], data=renda)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
plt.xticks( rotation=45);
plt.savefig('./output/figs/sexo.png')

plt.figure(figsize=(25, 18))
sns.countplot(x= renda['data_ref'],  hue = renda['posse_de_veiculo'], data=renda)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xticks( rotation=45);
plt.savefig('./output/figs/posse_de_veiculo.png')

plt.figure(figsize=(25, 18))
sns.countplot(x= renda['data_ref'],  hue = renda['posse_de_imovel'], data=renda)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xticks( rotation=45);
plt.savefig('./output/figs/posse_de_imovel.png')

plt.figure(figsize=(25, 18))
sns.countplot(x= renda['data_ref'],  hue = renda['tipo_renda'], data=renda)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xticks( rotation=45);
plt.savefig('./output/figs/tipo_renda.png')

plt.figure(figsize=(25, 18))
sns.countplot(x= renda['data_ref'],  hue = renda['educacao'], data=renda)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xticks( rotation=45);
plt.savefig('./output/figs/educacao.png')

plt.figure(figsize=(25, 18))
sns.countplot(x= renda['data_ref'],  hue = renda['estado_civil'], data=renda)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xticks( rotation=45);
plt.savefig('./output/figs/estado_civil.png')

plt.figure(figsize=(25, 18))
sns.countplot(x= renda['data_ref'],  hue = renda['tipo_residencia'], data=renda)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xticks( rotation=45);
plt.savefig('./output/figs/tipo_residencia.png')

# Plotando uma matriz de correção:

data_corr = renda.corr()
plt.figure(figsize=(25, 18))
sns.heatmap(data_corr, annot=True, fmt='.1g', vmin=-1, vmax=1, cmap='mako', center=0)
plt.xticks(rotation = 40)
#plt.show()
plt.savefig('./output/figs/correlacao.png')


# Identificando e tratando os dados missing:

renda.isna().sum()
renda.dropna(inplace=True)
renda.isna().sum()

X = pd.get_dummies(renda.drop(['Unnamed: 0', 'id_cliente', 'data_ref'], axis=1))
y = renda['renda']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state = 100)
print('X_train.shape', X_train.shape)
print('X_test.shape', X_test.shape)
print('y_train.shape', y_train.shape)
print('y_test.shape', y_test.shape)

regr_1 = DecisionTreeRegressor(max_depth=4)
regr_2 = DecisionTreeRegressor(max_depth=3)
regr_3 = DecisionTreeRegressor(max_depth=2)

regr_1.fit(X_train, y_train)
regr_2.fit(X_train, y_train)
regr_3.fit(X_train, y_train)

plt.rc('figure', figsize=(40, 10))
tp = tree.plot_tree(regr_1, 
                    feature_names=X.columns,  
                    filled=True)
plt.savefig('./output/figs/arvore_treino1.png')

plt.rc('figure', figsize=(10, 5))
tp = tree.plot_tree(regr_2, 
                    feature_names=X.columns,  
                    filled=True) 
plt.savefig('./output/figs/arvore_treino2.png')

plt.rc('figure', figsize=(10, 5))
tp = tree.plot_tree(regr_3, 
                    feature_names=X.columns,  
                    filled=True) 
plt.savefig('./output/figs/arvore_treino3.png')

mse1 = regr_1.score(X_train, y_train)
mse2 = regr_2.score(X_train, y_train)
mse2 = regr_3.score(X_train, y_train)

template = "O MSE da árvore de treino com profundidade={0} é: {1:.2f}"

print(template.format(regr_1.get_depth(),mse1).replace(".",","))
print(template.format(regr_2.get_depth(),mse2).replace(".",","))
print(template.format(regr_3.get_depth(),mse2).replace(".",","))

regr_1 = DecisionTreeRegressor(max_depth=4)
regr_2 = DecisionTreeRegressor(max_depth=3)
regr_3 = DecisionTreeRegressor(max_depth=2)

regr_1.fit(X_test, y_test)
regr_2.fit(X_test, y_test)
regr_3.fit(X_test, y_test)

plt.rc('figure', figsize=(40, 10))
tp = tree.plot_tree(regr_1, 
                    feature_names=X.columns,  
                    filled=True)
plt.savefig('./output/figs/arvore_teste1.png')

plt.rc('figure', figsize=(10, 5))
tp = tree.plot_tree(regr_3, 
                    feature_names=X.columns,  
                    filled=True) 
plt.savefig('./output/figs/arvore_teste2.png')

plt.rc('figure', figsize=(10, 5))
tp = tree.plot_tree(regr_3, 
                    feature_names=X.columns,  
                    filled=True) 
plt.savefig('./output/figs/arvore_teste3.png')

mse1 = regr_1.score(X_test, y_test)
mse2 = regr_2.score(X_test, y_test)
mse2 = regr_3.score(X_test, y_test)

template = "O MSE da árvore de teste com profundidade={0} é: {1:.2f}"

print(template.format(regr_1.get_depth(),mse1).replace(".",","))
print(template.format(regr_2.get_depth(),mse2).replace(".",","))
print(template.format(regr_3.get_depth(),mse2).replace(".",","))