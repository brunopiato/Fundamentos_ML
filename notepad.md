# Fundamentos de Machine Learning
_Comunidade DS - 07 de março_

<!-- TOC -->

- [Fundamentos de Machine Learning](#fundamentos-de-machine-learning)
  - [1. O que é Machine Learning e Inteligência Artificial](#1-o-que-é-machine-learning-e-inteligência-artificial)
  - [2. Tipos de aprendizado de máquina](#2-tipos-de-aprendizado-de-máquina)
  - [3. Aprendizado supervisionado](#3-aprendizado-supervisionado)
  - [4. K-Nearest Neighbors (KNN)](#4-k-nearest-neighbors-knn)
    - [4.1. Métricas de avaliação](#41-métricas-de-avaliação)
  - [5. Regressões e regressão linear](#5-regressões-e-regressão-linear)
    - [Regressão Linear](#regressão-linear)

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
<figcaption align = "center"><b>Fig. 1 - Exemplo de KNN em duas dimensões</b></figcaption>

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
<figcaption align = "center"><b>Fig. 3 - Matriz de confusão e tipos de erro</b></figcaption>

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
<figcaption align = "center"><b>Fig. 3 - Matriz de confusão e tipos de erro</b></figcaption>

<br>

- Estes modelos, bem como suas métricas, podem ser facilmente implementados em bibliotecas como _scikit-learn_, _stastmoldel_ e _scipy_.


## 5. Regressões e regressão linear
- No aprendizado supervisionado do tipo regressão estamos interessados em:
  - **Estudar o fenômeno** de interesse a fim de compreender a relação entre as variáveis e com elas afetam a variável-alvo ("variável-resposta" ou "variável-dependente")
  - **Elaborar um modelo matemático** capaz de prever o valor da variável-alvo com base nas variáveis-independentes ("variáveis-preditoras")
- Sendo a variável-resposta numérica
- Diversos algoritmos diferentes estimam o valor da variável-alvo, podendo ela ser a probabilidade de uma determinada classe e, assim, ser utilizados como algoritmos de classificação caso estipulemos um limiar de probabilidade.
  - Regressão linear
  - Regressão linear regulada (_lasso_ e _ridge_, ou _L1_ e _L2_)
  - Regressão polinomial
  - Regressão por redes neurais
  - Regressão de árvores de decisão
  - Regressão de florestas de aleatoriedade
  - Regressão por KNN
  - Regressão Gaussiana

### Regressão Linear
- Assumindo que a relação entre duas variáveis seja linear, ou seja, que variem proporcional e constantemente, modela a correlação entre estas variáveis
- Uma vez que as relações entre variáveis podem ser de dois tipos: **determinística** ou **estocástica**
  - **Determinística**: a relação entre as variáveis pode ser descrita com alta precisão e confiabilidade através de uma fórmula matemática (i.e. movimento retilíneo uniforme)
  - **Estocástica**: a relação entre as variáveis pode ser descrita probabilisticamente, incorporando um valor de erro e incerteza (i.e. lances de uma moeda)
- A regressão linear **descreve probabilisticamente** a relação entre duas ou mais variáveis
- Esta relação pode ser descrita por uma reta em um gráfico de coordenadas x e y (caso tenhamos apenas duas variáveis de interesse). Esta reta é descrita matematicamente por uma função (fórmula da reta), que apresenta **coeficiente linear (intercepto)** e **coeficiente angular (inclinação da reta)**, assumindo o formato

$$ y = a + bx $$

<br>

![Plano cartesiona reg linear](https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/LinearRegression.svg/1200px-LinearRegression.svg.png)
<figcaption align = "center"><b>Fig. 4 - Plano cartesiano da distribuição dos valores de duas variáveis e a reta de regressão linear</b></figcaption>

<br>

- O **ajuste do modelo de regressão linear** aos dados é feito pelo método de ajuste dos **mínimos quadrados**, encontrando os valores de _a_ e _b_ que determinam uma reta com o menor erro possível
- Uma vez estimados os valores dos coeficientes linear e angular, podemos usar a fórmula da reta para, sabendo o valor da variável _x_, estimar o valor da variável _y_
- A diferença entre os valores estimados e observados para a variável-alvo (_y_) é chamado de erro ou **resíduo da reta**. 
- O método dos quadrados mínimos encontra os valores dos coeficientes através dos valores do erro. A reta escolhida é aquela que tiver o **menor erro somado**. Contudo, como os erros podem acontecer por subestimativa ou superestimativa do valor de _y_, estes erros são elevados ao quadrado antes de serem somados para anular o efeito dos sinais positivos e negativos. Por este motivo este método de obtenção do valor da reta é chamado de **soma dos erros ao quadrado** (_Sum Square Error - SSE_)
- A regressão linear assume algumas **premissas**, que são:
  - A relação entre as variáveis é linear
  - Os erros são independentes
  - Os erros são normalmente distribuídos
  - Os erros para cada valor previsto tem variâncias iguais
- Existe uma série de meétricas de avaliação da regressão linear
  - O _SSE (Sum of Square Error)_ é a métrica de quão ajustada aos pontos a reta está
  - O _SSTO (Sum of Square Total)_ é a métrica de quão distantes da reta média estão os pontos
  - O _SSR (Sum of Square of Regression)_ é a métrica de quão distante a reta de regressão está em relação à reta média
  - Na imagem abaixo o _SSE_ pode ser compreendido como _RSS_; o _SSR_ pode ser compreendido como _ESS_; e o _SSTO_ pode ser compreendido como _TSS_

![Métricas da regressão linear](https://i.stack.imgur.com/LpHPy.png)
<figcaption align = "center"><b>Fig. 5 - Métricas da regressão linear</b></figcaption>

<br>

- A partir destes valores podemos estimar o valor de _R²_, que é uma métrica do quanto a reta de regressão é capaz de explocar os dados observador. O valor de _R²_ é dado pela fórmula:

$$ R²={1-{RSS \over TSS}} $$
- Ou seja, 1 menos a relação entre o quão ajustada a reta está aos pontos e o quão distante ela está em relação a reta média
- Podemos tirar algumas conclusões a partir da leitura desta fórmula 
  - Quando _RSS_ (_SSE_) e _TSS_ (_SSTO_) são muito próximas, a razão se aproxima de 1 e, ao ser subtraída de 1, resulta em um baixo valor de _R²_, indicando que a reta de regressão é tão boa quanto a reta da média, ou seja, a pior reta possível, para explicar os dados
  - Quanto maior o valor de _R²_, menor é o erro _RSS_ (_SSE_) em relação ao erro _TSS_ (_SSTO_), isto é, menor é o erro residual comparado com o erro da reta média e, portanto, melhor é a reta de regressão para explicar o fenômeno de interesse
- Contudo o valor de _R²_, é muito sensível a _outliers_, uma vez que as métricas usadas para calculá-lo são elevadas ao quadrado.
- Estes modelos, bem como suas métricas, podem ser facilmente implementados em bibliotecas como _scikit-learn_, _stastmoldel_ e _scipy_.
- Outras __métricas de performance__ do modelo podem ser aplicadas, como
  - _Erro médio quadrado_ (_Mean Squared Error - MSE_): calcula a média dos erros elevados ao quadrado entre os valores reais ou previstos. Sofre grande influencia de _outliers_ além de estar em escala quadrática em relação aos dados, uma vez que foi elevado ao quadrado. É chamado de função de perda (_loss function_) por ser uma forma de calcular o erro do modelo
  - _Erro médio absoluto_ (_Mean Absolute Error - MAE_): ao invés de usar o quadrado para anular o efeito dos sinais dos erros, utiliza o módulo, mantendo a escala da medida de erro igual à escala original dos dados, reduzindo também o efeito dos _outliers_
  - _Erro médio_ (_Mean Error - ME_): mantém o sinal dos erros ao calcular a média de modo que mantenhamos a informação de subestimação ou superestimação do modelos. Podemos saber para "qual lado" o modelo está errando mais
  - _Raiz do erro médio ao quadrado_ (_Root Square Mean Error - RMSE_): extrai a raiz do MSE para retornar o valor do erro para a mesma escala dos dados