# Fundamentos de Machine Learning
_Comunidade DS - 07 de março_

<!-- TOC -->

- [Fundamentos de Machine Learning](#fundamentos-de-machine-learning)
  - [1. O que é Machine Learning e Inteligência Artificial](#1-o-que-é-machine-learning-e-inteligência-artificial)
  - [2. Tipos de aprendizado de máquina](#2-tipos-de-aprendizado-de-máquina)
  - [3. Aprendizado supervisionado](#3-aprendizado-supervisionado)
  - [4. K-Nearest Neighbors (KNN)](#4-k-nearest-neighbors-knn)
    - [4.1. Métricas de avaliação](#41-métricas-de-avaliação)
  - [5. Regressão Linear](#5-regressão-linear)

<!-- /TOC -->

## 1. O que é Machine Learning e Inteligência Artificial
- __Inteligência artificial__ é uma ideia abstrata de um tipo de congnição computacional em que as máquinas um inteligência própria.
- __Aprendizado de máquina__ é a parte palpável da inteligência artificial. São algoritmos que permitem a máquinas tomarem decisões e aprender a partir de dados sem que sejam expressamente programadas para realizar uma dada tarefa. 
  - Utilizam o __método indutivo__ de aprendizado, isto é, a partir da observações repetidas de um dado fenômeno aprendem o padrão a ponto de conseguir tomar decisões sem que sejam expressamente programadas com *if*, *for* e *while* (programação tradicional).

## 2. Tipos de aprendizado de máquina
- __Aprendizado supervisionado:__ depende de um rótulo a ser aprendido com base em uma série de características apresentadas para o modelo. É utilizado em problemas de classificação, isto é, quando queremos que a máquina classifique novas entradas com base nas classificações apresentadas a ela durante seu treinamento. Podem ser utilizado também em problemas de regressão, isto é, quando o rótulo a ser aprendido é numérico e contínuo, e não categórico.
- __Aprendizado não-supervisionado:__ ????
- __Aprendizado semi-supervisionado:__ ????
- __Aprendizado por reforço:__ ????

## 3. Aprendizado supervisionado
- Depende de um rótulo a ser aprendido com base em uma série de características apresentadas para o modelo. É utilizado em problemas de classificação, isto é, quando queremos que a máquina classifique novas entradas com base nas classificações apresentadas a ela durante seu treinamento. Podem ser utilizado também em problemas de regressão, isto é, quando o rótulo a ser aprendido é numérico e contínuo, e não categórico.
- Existem muitos algoritmos que fazem classificação:
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
  - Light Gradient Boost Machine (LGBM)
  - Categorical Boost (CatBoost)
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Decision Tree
  - Random Forest
  - Neural Networks
- Existem vários algoritmos que fazem regressão:
  - Regressão linear
  - Regressão polinomial
- Alguns algoritmos de classificação recebem adaptações e também podem ser usados para problemas de regressão

## 4. K-Nearest Neighbors (KNN)
- É um algoritmo essencialmente de classificação com aprendizado supervisionado. Ele busca os K vizinhos mais próximos de um elemento para inferir a classificação mais provável deste elemento. Ele foi adaptado para ser utilizado também com problemas de regressão usando a média entre os valores dos K vizinhos, ao invés de uma votação de classe. 
- O KNN realiza um cálculo de distância entre a entrada a ser classificada e seus vizinhos em um hiperespaço n-dimensional para determinar quais são os K vizinhos mais próximos e, então realizar uma votação de classe entre eles (ou seja, buscar a moda do valor de classificação entre os K vizinhos).
  - Para tanto o algoritmo assume a premissa de que as variáveis escolhidas possam ser usadas para calcular distâncias entre elementos e são, portanto, numéricas.
- O KNN pode ser utilizado para:
  - Sistemas de recomendação
  - Classificação de notícias
  - Agrupamento de clientes
  - Classificação de imagens
  - Sistemas de busca
- O tempo de processamento é, entretanto, um fator limitante deste algoritmo, pois conforme o conjunto de dados aumenta em observações, aumenta também o número de cálculos de distância que o algoritmo deve executar. Outro fator limitante é a "maldição da dimensionalidade", isto é, conforme a dimensionalidade de um conjunto de dados cresce, aumenta também a distorção sobre os cálculos de distância entre os valores.

<br>

![Exemplo de KNN em duas dimensões](https://miro.medium.com/v2/resize:fit:1400/0*jqxx3-dJqFjXD6FA)
<figcaption align = "left"><b>Fig. 1 - Exemplo de KNN em duas dimensões</b></figcaption>

### 4.1. Métricas de avaliação
- É necessário avaliar se o algoritmo de fato aprendeu a executar as tarefas determinadas e, para isso, usamos as métricas de avaliação de aprendizado
- Existem diversas métricas e muitas delas são específicas para dados algoritmos ou tipos de aprendizado.
  - Matriz de confusão
  - Acurácia
  - Precisão (_precision_)
  - Retorno (_recall_)
- __Matriz de confusão__ é uma tabela de contingência que tabula os falsos positivos, falsos negativos, verdadeiros positivos e verdadeiros negativos, de acordo com as classes que temos no conjunto de dados.

![Matriz de confusão](https://www.dataschool.io/content/images/2015/01/confusion_matrix2.png)
<figcaption align = "left"><b>Fig. 2 - Matriz de confusão</b></figcaption>

- Na matriz de confusão acima temos:
  - Nas linhas os valores observados
  - Nas colunas os valores preditos
  - Na primera linha temos o número real de NO (n=60)
  - Na segunda linha temos o número real de YES (n=105)
  - Na primeira coluna temos o número predito de NO (n=55)
  - Na segunda coluna temos o número predito de YES (n=110)
  - No primeiro quadrante temos os verdadeiros negativos (TN=50)
  - No segundo quadrante temos os falsos negativos (FN=5)
  - No terceiro quadrante temos os verdadeiros positivos (TP=105)
  - No quarto quadrante temos os falsos ppositivos (FP=10)
  - A soma das duas linhas tem que ser igual à soma das duas colunas
- A partir desta matriz podemos calcular algumas medidas:
  - __Acurácia:__ o número de acertos do valor predito em relação ao total de tentativas _(TN+TP/PredNO+PredYes)_. Neste caso: _50+100/55+110=0.909_
  - Como a acurácia nem sempre é uma boa medida. Principalmente quando os conjuntos de dados estão desbalanceados (muito mais ocorrências de uma classe do que da outra), podemos usar outras medidas, como precisão e retorno.

<br>

  - ![Matriz de confusão e tipos de erro](https://miro.medium.com/v2/resize:fit:667/1*3yGLac6F4mTENnj5dBNvNQ.jpeg)
<figcaption align = "left"><b>Fig. 3 - Matriz de confusão e tipos de erro</b></figcaption>

<br>

  - __Precisão:__ é a quantidade de acertos da classe em relação à quantidade de vezes que o algoritmo afirmou aquela classificação _(TP/TP+FP)_
    - A precisão indica o quão preciso em afirmar uma classe o algoritmo é
  - __Retorno:__ é a quantidade de vezes que o algoritmo acertou a classificação em relação ao total de ocorrências daquela classe _(TP/TP+FN)_
    - O retorno indica a porcentagem de ocorrências de uma classe o algoritmo conseguiu recuperar do conjunto de dados
    - Quando estamos estimando o retorno de uma classe positiva, chamamos de sensibilidade (_sensitivity_)
    - Quando estamos estimando o retorno de uma classe negativa, chamamos de especificidade (_specificity_)
    - Estes nomes vem da epidemiologia e por isso, por vezes não se aplicam com precisão aos problemas de aprendizado de máquina

<br>

$$ Acurácia = {TP + TN \over ( TP + TN + FP + FN)} $$
$$ Precisão = {TP \over ( TP + FP )} $$
$$ Retorno = {TP \over ( TP + FN)} $$
$$ F1Score = {2TP \over ( 2TP + FP + FN)} $$

<br>

![Métricas de avaliação a partir da matriz de confusão](https://2.bp.blogspot.com/-EvSXDotTOwc/XMfeOGZ-CVI/AAAAAAAAEiE/oePFfvhfOQM11dgRn9FkPxlegCXbgOF4QCLcBGAs/s1600/confusionMatrxiUpdated.jpg)
<figcaption align = "left"><b>Fig. 3 - Matriz de confusão e tipos de erro</b></figcaption>

<br>


## 5. Regressão Linear


