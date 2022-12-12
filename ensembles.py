# coding=utf-8
from time import time #Importamos la función time para capturar tiempos


tiempo_inicial = time()

import numpy as np #Extensión de Python que le agrega mayor soporte para vectores y matrices de funciones matemáticas de alto nivel
import dataset as ds # Extensión para lectura de Datasets
import matplotlib.pyplot as plt #Proporciona un marco de graficado similar a MATLAB
from os import walk #Genera los nombres de los archivos en un árbol de directorios
from scipy import stats #Genera todas las funciones de estadísticas de la libreria Scipy
from sklearn import svm


#Scikit-learn librería especializada en algoritmos para data mining y machine learning

from sklearn import metrics #Consulta la evaluación del modelo: cuantificación de la calidad de las predicciones y la sección métricas
from sklearn.ensemble import BaggingClassifier #Extensión que permite generar un conjunto meta-estimador que se ajusta a los clasificadores básicos en subconjuntos aleatorios del conjunto de datos original y luego agrega sus predicciones individuales (ya sea votando o promediando) para formar una predicción final.
from sklearn.linear_model import Perceptron #  Algoritmo para el aprendizaje supervisado de clasificadores binarios de tipo lineal
from sklearn.neighbors import KNeighborsClassifier #Extensión para usar el clasificador que implementa k vecinos mas cercanos KNN.
from sklearn.decomposition import PCA # Extensión ( Principal Component Analysis) que permite descomponer un conjunto de datos multivariado en un conjunto de componentes ortogonales
from imblearn.under_sampling import EditedNearestNeighbours #Clase para realizar un submuestreo basado en el método del vecino más cercano editado
from imblearn.over_sampling import SMOTE


from sklearn.utils import shuffle #Mezcla matrices de forma consistente (Mezclar Dataset)



#Método para lectura de Dataset con modelo de validación cruzada de k particiones (k-fcv)
def datasets(ruta):
    total = []
    for (path, folder, file) in walk(ruta):
        for i in range(len(file)):
            if i%2==1:
                dt.append(ruta+"/"+file[i])
                total.append(dt)
            else:
                dt = []
                dt.append(ruta+"/"+file[i])
    return total

#____________________________________________________
#             DataSets de clasificación estándar
#____________________________________________________

#dataSets = datasets('./DataSets/appendicitis-5-fold')
#dataSets = datasets('./DataSets/australian-5-fold')
#dataSets = datasets('./DataSets/automobile-5-fold')
#dataSets = datasets('./DataSets/balance-5-fold')
#dataSets = datasets('./DataSets/breast-5-fold')
#dataSets = datasets('./DataSets/bupa-5-fold')
#dataSets = datasets('./DataSets/car-5-fold')
#dataSets = datasets('./DataSets/chess-5-fold')
#dataSets = datasets('./DataSets/cleveland-5-fold')
#dataSets = datasets('./DataSets/contraceptive-5-fold')
#dataSets = datasets('./DataSets/crx-5-fold')
#dataSets = datasets('./DataSets/dermatology-5-fold')
#dataSets = datasets('./DataSets/ecoli-5-fold')
#dataSets = datasets('./DataSets/flare-5-fold')
#dataSets = datasets('./DataSets/glass-5-fold')
#dataSets = datasets('./DataSets/haberman-5-fold')
#dataSets = datasets('./DataSets/hayes-roth-5-fold')
#dataSets = datasets('./DataSets/heart-5-fold')
#dataSets = datasets('./DataSets/iris-5-fold')
#dataSets = datasets('./DataSets/led7digit-5-fold')
#dataSets = datasets('./DataSets/lymphography-5-fold')
#dataSets = datasets('./DataSets/mammographic-5-fold')
#dataSets = datasets('./DataSets/monk-2-5-fold')
#dataSets = datasets('./DataSets/movement_libras-5-fold')
#dataSets = datasets('./DataSets/newthyroid-5-fold')
#dataSets = datasets('./DataSets/pima-5-fold')
#dataSets = datasets('./DataSets/saheart-5-fold')
#dataSets = datasets('./DataSets/segment-5-fold')
#dataSets = datasets('./DataSets/sonar-5-fold')
#dataSets = datasets('./DataSets/spectfheart-5-fold')
#dataSets = datasets('./DataSets/splice-5-fold')
#dataSets = datasets('./DataSets/tae-5-fold')
#dataSets = datasets('./DataSets/tic-tac-toe-5-fold')
#dataSets = datasets('./DataSets/titanic-5-fold')
#dataSets = datasets('./DataSets/vehicle-5-fold')
#dataSets = datasets('./DataSets/vowel-5-fold')
#dataSets = datasets('./DataSets/wine-5-fold')
#dataSets = datasets('./DataSets/wisconsin-5-fold')
#dataSets = datasets('./DataSets/yeast-5-fold')
#dataSets = datasets('./DataSets/zoo-5-fold')

#____________________________________________________
#             DataSets no balanceados para clasificación
#____________________________________________________

dataSets = datasets('./DataSets_IR/glass1-5-fold')
#dataSets = datasets('./DataSets_IR/ecoli-0_vs_1-5-fold')
#dataSets = datasets('./DataSets_IR/wisconsin-5-fold')
#dataSets = datasets('./DataSets_IR/pima-5-fold')
#dataSets = datasets('./DataSets_IR/iris0-5-fold')
#dataSets = datasets('./DataSets_IR/glass0-5-fold')
#dataSets = datasets('./DataSets_IR/yeast1-5-fold')
#dataSets = datasets('./DataSets_IR/haberman-5-fold')
#dataSets = datasets('./DataSets_IR/vehicle2-5-fold')
#dataSets = datasets('./DataSets_IR/vehicle1-5-fold')
#dataSets = datasets('./DataSets_IR/vehicle3-5-fold')
#dataSets = datasets('./DataSets_IR/glass-0-1-2-3_vs_4-5-6-5-fold')
#dataSets = datasets('./DataSets_IR/vehicle0-5-fold')
#dataSets = datasets('./DataSets_IR/ecoli1-5-fold')
#dataSets = datasets('./DataSets_IR/new-thyroid1-5-fold')
#dataSets = datasets('./DataSets_IR/new-thyroid2-5-fold')
#dataSets = datasets('./DataSets_IR/ecoli2-5-fold')
#dataSets = datasets('./DataSets_IR/segment0-5-fold')
#dataSets = datasets('./DataSets_IR/glass6-5-fold')
#dataSets = datasets('./DataSets_IR/yeast3-5-fold')
#dataSets = datasets('./DataSets_IR/ecoli3-5-fold')
#dataSets = datasets('./DataSets_IR/page-blocks0-5-fold')
#dataSets = datasets('./DataSets_IR/ecoli-0-3-4_vs_5-5-fold')
#dataSets = datasets('./DataSets_IR/yeast-2_vs_4-5-fold')
#dataSets = datasets('./DataSets_IR/ecoli-0-6-7_vs_3-5-5-fold')
#dataSets = datasets('./DataSets_IR/ecoli-0-2-3-4_vs_5-5-fold')
#dataSets = datasets('./DataSets_IR/yeast-0-3-5-9_vs_7-8-5-fold')
#dataSets = datasets('./DataSets_IR/glass-0-1-5_vs_2-5-fold')
#dataSets = datasets('./DataSets_IR/yeast-0-2-5-7-9_vs_3-6-8-5-fold')
#dataSets = datasets('./DataSets_IR/yeast-0-2-5-6_vs_3-7-8-9-5-fold')
#dataSets = datasets('./DataSets_IR/ecoli-0-4-6_vs_5-5-fold')
#dataSets = datasets('./DataSets_IR/ecoli-0-1_vs_2-3-5-5-fold')
#dataSets = datasets('./DataSets_IR/ecoli-0-2-6-7_vs_3-5-5-fold')
#dataSets = datasets('./DataSets_IR/glass-0-4_vs_5-5-fold')
#dataSets = datasets('./DataSets_IR/ecoli-0-3-4-6_vs_5-5-fold')
#dataSets = datasets('./DataSets_IR/ecoli-0-3-4-7_vs_5-6-5-fold')
#dataSets = datasets('./DataSets_IR/yeast-0-5-6-7-9_vs_4-5-fold')
#dataSets = datasets('./DataSets_IR/vowel0-5-fold')
#dataSets = datasets('./DataSets_IR/ecoli-0-6-7_vs_5-5-fold')
#dataSets = datasets('./DataSets_IR/glass-0-1-6_vs_2-5-fold')
#dataSets = datasets('./DataSets_IR/ecoli-0-1-4-7_vs_2-3-5-6-5-fold')
#dataSets = datasets('./DataSets_IR/led7digit-0-2-4-5-6-7-8-9_vs_1-5-fold')
#dataSets = datasets('./DataSets_IR/glass-0-6_vs_5-5-fold')
#dataSets = datasets('./DataSets_IR/ecoli-0-1_vs_5-5-fold')
#dataSets = datasets('./DataSets_IR/glass-0-1-4-6_vs_2-5-fold')
#dataSets = datasets('./DataSets_IR/glass2-5-fold')
#dataSets = datasets('./DataSets_IR/ecoli-0-1-4-7_vs_5-6-5-fold')
#dataSets = datasets('./DataSets_IR/ecoli-0-1-4-6_vs_5-5-fold')
#dataSets = datasets('./DataSets_IR/cleveland-0_vs_4-5-fold')
#dataSets = datasets('./DataSets_IR/shuttle-c0-vs-c4-5-fold')
#dataSets = datasets('./DataSets_IR/yeast-1_vs_7-5-fold')
#dataSets = datasets('./DataSets_IR/glass4-5-fold')
#dataSets = datasets('./DataSets_IR/ecoli4-5-fold')
#dataSets = datasets('./DataSets_IR/page-blocks-1-3_vs_4-5-fold')
#dataSets = datasets('./DataSets_IR/glass-0-1-6_vs_5-5-fold')
#dataSets = datasets('./DataSets_IR/shuttle-c2-vs-c4-5-fold')
#dataSets = datasets('./DataSets_IR/yeast-1-4-5-8_vs_7-5-fold')
#dataSets = datasets('./DataSets_IR/glass5-5-fold')
#dataSets = datasets('./DataSets_IR/yeast-2_vs_8-5-fold')
#dataSets = datasets('./DataSets_IR/yeast4-5-fold')
#dataSets = datasets('./DataSets_IR/yeast-1-2-8-9_vs_7-5-fold')
#dataSets = datasets('./DataSets_IR/yeast5-5-fold')
#dataSets = datasets('./DataSets_IR/ecoli-0-1-3-7_vs_2-6-5-fold')
#dataSets = datasets('./DataSets_IR/yeast6-5-fold')



#Arrays para almacenar los resultados de cada particion del dataset,obteniendo el score de prediccion de las tecnicas de Selección Dinamica (DS) frente al y_test
accuracyOLA = []
accuracyLCA = []
accuracyMV =[]
accuracyWV = []

#Arrays para almacenar los resultados de cada particion del dataset,obteniendo el score de prediccion de las tecnicas de Selección Dinamica (DS) utilizando el algoritmo (Dynamic Frienemy Pruning) DFP (filtrado de Clasificadores) frente al y_test
accuracyOLApruned = []
accuracyLCApruned = []
accuracyMVpruned = []
accuracyWVpruned = []

#__________________________________________________________________________________________
# 1. Superproducción (Bagging)
#__________________________________________________________________________________________

# Bagging
#Entrada: Los datasets X_train,y_train,X_test,y_test
#Salida: Retornamos el conjunto de 100 perceptrones y el numero de clases existente en el dataset
def bagging(X_train, y_train, X_test, y_test):
    bag = BaggingClassifier(Perceptron(), n_estimators=100, random_state=12) #Generamos 100 perceptrones con semilla de 12
    bag.fit(X_train, y_train) #Entrenamos el conjunto de clasificadores
    pr = bag.predict(X_test) #Predecimos con respecto al X_test
    score = metrics.accuracy_score(pr, y_test) * 100 #Estimamos su score
    #print score
    return bag.estimators_,bag.n_classes_


#__________________________________________________________________________________________
#2. Fase de Filtrado: ENN
#__________________________________________________________________________________________

# Función para gráficar datasets 3 clases
def plot_resampling3(ax, X, y, title):
    c0 = ax.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0", alpha=0.5)
    c1 = ax.scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1", alpha=0.5)
    c2 = ax.scatter(X[y == 2, 0], X[y == 2, 1], label="Class #2", alpha=0.5)
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlim([-3, 7])
    ax.set_ylim([-3, 2])
    return c0, c1, c2

# Función para gráficar datasets 2 clases
def plot_resampling2(ax, X, y, title):
    c0 = ax.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0", alpha=0.5)
    c1 = ax.scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1", alpha=0.5)
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlim([-0.6, 0.8])
    ax.set_ylim([-0.4, 0.8])
    return c0, c1

# Método para graficar ejemplos removidos debido a función ENN
def plotENN_3clases(X_vis, X_res_vis, y_resampled, reduction_str, idx_samples_removed):
    f, (ax1, ax2) = plt.subplots(1, 2)
    c0, c1, c2 = plot_resampling2(ax1, X_vis, y_train, 'Original dataset')
    c3 = ax2.scatter(X_vis[idx_samples_removed, 0], X_vis[idx_samples_removed, 1], alpha=.2, label='Ejemplos removidos',
                     c='k', marker='_')
    plot_resampling2(ax2, X_res_vis, y_resampled, 'ENN - ' + reduction_str)
    plt.figlegend((c0, c1, c2, c3), ('Clase #0', 'Clase #1', 'Clase #2', 'Ejemplos removidos'),
                  loc='lower center', ncol=3, labelspacing=0.)
    plt.tight_layout(pad=3)
    plt.show()

def plotENN_2clases(X_vis, X_res_vis, y_resampled, reduction_str, idx_samples_removed):
    f, (ax1, ax2) = plt.subplots(1, 2)
    c0, c1 = plot_resampling2(ax1, X_vis, y_train, 'Original dataset')
    c3 = ax2.scatter(X_vis[idx_samples_removed, 0], X_vis[idx_samples_removed, 1], alpha=.2, label='Ejemplos removidos',
                     c='k', marker='_')
    plot_resampling2(ax2, X_res_vis, y_resampled, 'ENN - ' + reduction_str)
    plt.figlegend((c0, c1, c3), ('Clase #0', 'Clase #1', 'Ejemplos removidos'),
                  loc='lower center', ncol=3, labelspacing=0.)
    plt.tight_layout(pad=3)
    plt.show()

def plotENN_2clases_SMOTE(X_vis, X_res_vis, y_resampled, reduction_str):
    f, (ax1, ax2) = plt.subplots(1, 2)
    c0, c1 = plot_resampling2(ax1, X_vis, y_train, 'Original dataset')
    plot_resampling2(ax2, X_res_vis, y_resampled, 'ENN' + reduction_str)
    plt.figlegend((c0, c1), ('Clase #0', 'Clase #1', 'Ejemplos Aumentados'),
                  loc='lower center', ncol=3, labelspacing=0.)
    plt.tight_layout(pad=3)
    plt.show()

#Método ENN: Técnica de Selección conocida por su eficiencia para eliminar el ruido y producir límites de clases más suaves.
def ENN(X_train,y_train):
    # Crea una instancia de un objeto de PCA para facilitar la visualización
    pca = PCA(n_components=3)
    # Ajustar y transformar x para visualizar dentro de un espacio de características 2D
    X_vis = pca.fit_transform(X_train)
    # Aplicamos algoritmo de ENN
    enn = EditedNearestNeighbours(return_indices=True)
    # Obtenemos X, y y los indices del dataset ya remuestreado
    X_resampled, y_resampled, idx_resampled = enn.fit_sample(X_train, y_train)
    #X_res_vis = pca.transform(X_resampled)
    #idx_samples_removed = np.setdiff1d(np.arange(X_vis.shape[0]), idx_resampled)
    #reduction_str = ('ENN Removio {:.2f}%'.format(100 * (1 - float(len(X_resampled)) / len(X_train))))
    #print(reduction_str) # Ver % de muestras removidas
    #plotENN_2clases(X_vis,X_res_vis, y_resampled, reduction_str, idx_samples_removed) #Opcional ver gráfica
    return X_resampled, y_resampled

#Aplica SMOTE over-sampling
def ENN2(X_train,y_train):
    # Crea una instancia de un objeto de PCA para facilitar la visualización
    pca = PCA(n_components=3)
    # Ajustar y transformar x para visualizar dentro de un espacio de características 2D
    #X_vis = pca.fit_transform(X_train)
    sm = SMOTE(ratio = 'auto', kind='regular')
    X_resampled, y_resampled = sm.fit_sample(X_train, y_train)
    #X_res_vis = pca.transform(X_resampled)
    #aumento_str = (' Aumento {:.2f}%'.format(-100 * (1 - float(len(X_resampled)) / len(X_train))))
    #idx_samples_removed = np.setdiff1d(np.arange(X_vis.shape[0]), idx_resampled)
    #plotENN_2clases_SMOTE(X_vis, X_res_vis, y_resampled, aumento_str)  # Opcional ver gráfica
    return X_resampled, y_resampled

#__________________________________________________________________________________________
#3. Fase de Región de Competencia: KNN & KNNE
#__________________________________________________________________________________________

#__________________________________________________________________________________________
####################               KNN K-Nearest Neighbors          #######################
#__________________________________________________________________________________________


#K-Nearest Neighbors
    #Entrada: dataset X_tra, y_tra, k
    #Salida: indices de los ejemplos, predicciones y el score de exactitud
def KNN(X_tra, y_tra, k):
    # Llamamos al constructor de KNN
    KNN = KNeighborsClassifier(n_neighbors=k, weights='uniform', metric='minkowski', p=2)
    # Entrenamos KNN
    clasif = KNN.fit(X_tra, y_tra)
    idxEjemplos = KNN.kneighbors(X_tra, return_distance=False)
    # Realizamos las predicciones para los pixeles de entrenamiento
    predicciones = clasif.predict(X_tra)
    #print predicciones,predicciones.shape
    acc = metrics.accuracy_score(y_tra, predicciones) * 100
    # Mostramos la matriz de confusión
    #cm = metrics.confusion_matrix(y_tra, predicciones)
    #print cm
    # Obtenemos las clases de los indices
    clases = y_tra[idxEjemplos]
    return idxEjemplos,clases,acc

#__________________________________________________________________________________________
####################         KNN K-Nearest Neighbors Equiality      #######################
#__________________________________________________________________________________________



#K-Nearest Neighbors Equality
    #Entrada: dataset X_tra, y_tra, k
    #Salida: indices de los ejemplos, predicciones y el score de exactitud
def KNNE(X_tra, y_tra, k):

    tam = y_tra.shape[0] # Numero de Ejemplos
    clasesDataSet = list(set(y_tra)) #Lista de clases del dataset
    distanciaMediaPorClase = []  # Diccionario para guardar las distancia promedio por cada clase
    distEjemplos = []
    idxEjemplos = []

    for clase_actual in clasesDataSet:

        mask = (y_tra == clase_actual)
        numEjemplosPorClase = len(y_tra[mask])

        if  k <= numEjemplosPorClase:
            # Llamamos al constructor de KNN
            KNN = KNeighborsClassifier(n_neighbors=k, weights='uniform', metric='minkowski', p=2)
        else:
            # Llamamos al constructor de KNN
            KNN = KNeighborsClassifier(n_neighbors=numEjemplosPorClase, weights='uniform', metric='minkowski', p=2)

        # Entrenamos KNN
        KNN = KNN.fit(X_tra[mask], y_tra[mask])
        # Obtenemos las distancias y los indices de cada instancia
        dist_idx = KNN.kneighbors(X_tra, return_distance=True)
        distanciaEjemplos = dist_idx[0]
        distEjemplos.append(distanciaEjemplos) #Agrego las distancias de todos los ejemplos

        # Relacionamos los ínidces que te devuelve el método con las posiciones donde y_tra == clase_actual sea True (índices absolutos y no relativos)
        idxKneighbors = dist_idx[1]  # Indices de retorno del kneighbors
        correspondencia = np.where(mask)[0]
        idxReferencia = correspondencia[idxKneighbors]
        idxEjemplos.append(idxReferencia)

        #Calcular la distancia promedio de cada ejemplo por cada clase
        #Utilizaremos distanciaEjemplos e idxReferencia para su cálculo
        distanciaMediaPorClase.append(np.mean(distanciaEjemplos,axis=1))

    # Realizamos las predicciones para los pixeles de entrenamiento y obtenemos con ellos los indices de las clases predichas
    pr = np.argmin(distanciaMediaPorClase,axis=0)
    acc = metrics.accuracy_score(y_tra, pr) * 100

    # Concatenamos los indices absolutos
    idxConcatenate = idxEjemplos[0]
    for i in np.arange(1,len(clasesDataSet)):
        idxConcatenate = np.hstack((idxConcatenate, idxEjemplos[i]))

    # Mostramos la matriz de confusión
    #cm = metrics.confusion_matrix(y_tra, pr)
    #print cm

    # Obtenemos las clases de los indices
    clases = y_tra[idxConcatenate]
    return idxConcatenate, clases, acc



#Aplicamos las fases, N° 2: Fase de Filtrado (ENN)  & N° 3: Fase de Region de competencia (KNN o KNNE)
    #Entradas: datasets X_train, y_train, k, flagENN, flagKNNE
    #Salidas: indices de las instancias más cercanas, las clases y los datasets X_tra e y_tra remuestreado
def FilteringRoC(X_train, y_train, k, flagENN, flagKNNE):

    #Desordenar Dataset
    #X_train, y_train = shuffle(X_train, y_train, random_state=0)

    # flagENN: True si aplicamos el algoritmo ENN , false sin aplicar algoritmo ENN
    if flagENN == True:
        X_tra, y_tra = ENN(X_train, y_train)
    else:
        X_tra, y_tra = X_train, y_train

        # flagKNNE: True si aplicamos el algoritmo KNNE, false aplicamos KNN
    if flagKNNE == True:
        idxEjemplos, clases, acc = KNNE(X_tra, y_tra,k)
    else:
        idxEjemplos, clases, acc = KNN(X_tra, y_tra,k)

    #Verificación
    if ((flagENN == True) & (flagKNNE == True)):
        print 'Con ENN & Con KNNE: '+ str(acc)
    elif((flagENN == True) & (flagKNNE == False)):
        print 'Con ENN & Con KNN: ' + str(acc)
    elif ((flagENN == False) & (flagKNNE == True)):
        print 'Sin ENN & Con KNNE: ' + str(acc)
    else:
        print 'Sin ENN & Con KNN: ' + str(acc)

    return idxEjemplos,clases,X_tra,y_tra

#__________________________________________________________________________________________
#4. Fase de Selección: Dynamic Frienemy Pruning (DFP) & Dynamic Selection (DS)
#__________________________________________________________________________________________


#__________________________________________________________________________________________
################     Tecnicas de Selección Dinámica (DS) (OLA, LCA, MV, WV) ###############
#__________________________________________________________________________________________

#Overal Local Accuracy (OLA)
#Entrada: Conjunto de C, los datasets X_train,y_train,X_test,y_test, y el numero de vecinos K
#Salida: El mejor clasificador para cada prueba de test
def ola(C, X_train,y_train,X_test,y_test,K):
        KNN = KNeighborsClassifier(n_neighbors=K)
        KNN.fit(X_train,y_train)
        salidastst = np.zeros_like(y_test)
        #Para cada ejemplo de X_test
        for i in range(X_test.shape[0]):
            salidas = map(lambda clasif: clasif.predict(X_test[i,:].reshape(1,-1)), C)
            #Si todos los clasificadores tienen la misma etiqueta
            if((salidas[0] == salidas).all()):
                salidastst[i] = salidas[0]
            else:
                # retornar indice K mas cercano al train
                idxEjemplos = KNN.kneighbors(X_test[i, :].reshape(1, -1), return_distance=False)[0]
                aciertosMax = -1
                clasifFinal = None
                for c in C:
                    #Para cada clasificador calcular el OLA como el porcentaje de clasificacion correcta de cada clasificador para cada indice
                    salidas = c.predict(X_train[idxEjemplos,:])
                    aciertos = np.sum(salidas == y_train[idxEjemplos])
                    if(aciertos > aciertosMax): # Selecciona el mejor clasificador
                        aciertosMax = aciertos
                        clasifFinal = c
                salidastst[i] = clasifFinal.predict(X_test[i,:].reshape(1,-1))
        return salidastst

# Local Class Accuracy (LCA)
#Entrada: Conjunto de C, los datasets X_train,y_train,X_test,y_test, y el numero de vecinos K
#Salida: El mejor clasificador para cada prueba de test
def lca(C, X_train,y_train,X_test,y_test,K):
    KNN = KNeighborsClassifier(n_neighbors=K)
    KNN.fit(X_train,y_train)
    salidastst = np.zeros_like(y_test)
    for i in range(X_test.shape[0]):
        salidas = map(lambda clasif: clasif.predict(X_test[i,:].reshape(1,-1)), C)
        # Si todos los clasificadores tienen la misma etiqueta
        if((salidas[0] == salidas).all()):
            salidastst[i] = salidas[0]
        else:
            aciertosMax = -1
            clasifFinal = None

            for c in C:
                # wj = ci(t) La salida predicha del clasificador ci para cada ejemplo
                w = c.predict(X_test[i,:].reshape(1,-1))
                # retornar indice K mas cercano al train a lo largo de la clase wj
                idxEjemplos = KNN.kneighbors(X_test[i, :].reshape(1, -1), return_distance=False)[0]
                mask = y_train[idxEjemplos] == w
                if (np.sum(mask) == 0):
                    continue
                # Calcular el LCA como el porcentaje de ejemplos etiquetados correctamente de wj por cada clasificador ci para cada indice
                salidas = c.predict(X_train[idxEjemplos][mask, :])
                aciertos = float(np.sum(salidas == y_train[idxEjemplos][mask])) / float(np.sum(mask))
                if(aciertos > aciertosMax):
                    aciertosMax = aciertos
                    clasifFinal = c
                salidastst[i] = clasifFinal.predict(X_test[i, :].reshape(1, -1))
    return salidastst

# Majority Voting
#Entrada: Conjunto de C, los datasets X_test
#Salida: El mejor clasificador para cada prueba de test
def majorityVoting(C,X_test):
    salidas = np.zeros( (len(X_test),len(C)) )
    for i,c in enumerate(C):
        salidas[:, i] = c.predict(X_test)
    pred = stats.mode(salidas,axis=1)#Selecciona la clase con mayor voto
    #score = metrics.accuracy_score(pr, y_test) * 100
    pr = list(map(int,(pred[0].T)[0].tolist()))
    return  pr

# Weigth Voting
#Entrada: Conjunto de C,numero de clases, los datasets X_test
#Salida: El mejor clasificador para cada prueba de test
def weigthVoting(C,num_clases,X_test):
    pr = []
    if num_clases != 2:
        salidas = np.zeros( (len(X_test),num_clases) )
    else:
        salidas = np.zeros((len(X_test)))

    for i,c in enumerate(C):
        salidas = salidas + c.decision_function(X_test)

    if num_clases != 2:
        pr = np.argmax(salidas,axis=1) #Selecciona la clase con mayor peso
    else:
        pr = list(map(lambda x: 0 if x < 0 else 1, salidas))

    #score = metrics.accuracy_score(pr, y_test) * 100
    return pr

#__________________________________________________________________________________________
###########################     Dynamic Frienemy Pruning (DFP) ############################
#__________________________________________________________________________________________

#Método para obtener todos los pares frienemies (Parejas de diferente clase)
def frienemy(idx,clases):
    # Variable para agregar todas los pares frienemies
    totalFrienemy = []
    numEjemplos = len(clases)
    for i in np.arange(numEjemplos):#itero por numero de ejemplos
        frienemy = []
        #Verifico si todos los elementos de cada ejemplo de la clase son iguales, si es True, no existe frienemies
        flag = all([ x==clases[i][0] for x in clases[i] ])
        if (flag == False):#Por cada ejemplo existe dos o más clases diferentes
            tamEjemplo = len(idx[i])
            for n  in np.arange(tamEjemplo):#itero por numero de elemento de cada ejemplo
                for m in np.arange(tamEjemplo):#itero por numero de elemento de cada ejemplo
                    if((clases[i][n]!=clases[i][m])):#Compruebo si son clases iguales
                        par = list(np.sort([idx[i][n],idx[i][m]]))#genero el frienemie
                        if(par not in totalFrienemy):
                            totalFrienemy.append(par)#lo agrego a la lista
    return totalFrienemy

def DFP(C, idx,clases,X_tra,y_tra):
    Cpruned = []
    F = frienemy(idx,clases)
    for c in C:
        for frienemies in F:
            #Pares correctamente clasificados
            x0 = c.predict(X_tra[frienemies[0],:].reshape(1, -1))[0]
            x1 = c.predict(X_tra[frienemies[1],:].reshape(1, -1))[0]
            y0 = y_tra[frienemies[0]]
            y1 = y_tra[frienemies[1]]
            if ((x0==y0) & (x1== y1)):
                Cpruned.append(c)
                break
    if len(Cpruned) == 0:
        Cpruned = C

    tam_Cpruned = len(Cpruned)
    print "C_eliminados: "+str(len(C)-tam_Cpruned),";", "C_para_analizar: "+str(tam_Cpruned)
    return Cpruned

def media_aritmetica(matrix_confusion):
    n_clases = len(matrix_confusion);
    am = 0;
    for i in np.arange(n_clases):
        am += float(matrix_confusion[i,i]) / sum(matrix_confusion[i,:])
    am = float(am / n_clases)
    return am
#__________________________________________________________________________________________
########################     Ejecución FIRE-DES++    ######################################
#__________________________________________________________________________________________

for i in dataSets:
    datosTrain, _ = ds.lecturaDatos(i[0]) #Datos de Entrenamiento
    datosTest, _ = ds.lecturaDatos(i[1]) #Datos de Prueba
    k = 7 #Vecinos a analizar por cada instancia
    X_train, y_train = datosTrain.data, datosTrain.target
    X_test, y_test = datosTest.data, datosTest.target

    #### Clasificadores
    #Aplicamos la fase N° 1: Fase Superproducción (Bagging)
        # Entrada: Datasets X_train,y_train,X_test,y_test
        # Salida: Obtenemos el conjunto de clasificadores (100 perceptrones) y numero de clases
    C_bag = bagging(X_train,y_train,X_test,y_test)

    # Conjunto de clasificadores
    C = C_bag[0]
    # Numero de clases en total
    n_clases = C_bag[1]

    #Aplicamos las fases, N° 2: Fase de Filtrado (ENN)  & N° 3: Fase de Region de competencia (KNN o KNNE)
        #Entradas: datasets X_train, y_train, k, flagENN, flagKNNE
            # flagENN: True si aplicamos el algoritmo ENN , false sin aplicar algoritmo ENN
            # flagKNNE: True si aplicamos el algoritmo KNNE, false aplicamos KNN
        #Salidas: indices de las instancias más cercanas, las clases y los datasets X_tra e y_tra remuestreado
    idx, clases, X_tra, y_tra = FilteringRoC(X_train, y_train, k, flagENN = False , flagKNNE = False)

    #Aplicamos Fase N° 4: Selección
    # DFP: Dynamic Frienemy Pruning
    Cpruned = DFP(C, idx, clases, X_tra, y_tra)

    #Predecimos las técnicas de selección dinámica (OLA, LCA, MV, WV) por cada iteración
    prediccionesOLA = ola(C, X_train, y_train, X_test, y_test,k)
    prediccionesLCA = lca(C, X_train, y_train, X_test, y_test, k)
    prediccionesMV = majorityVoting(C, X_test)
    prediccionesWV = weigthVoting(C,n_clases, X_test)

    #Predecimos las técnicas de selección dinámica (OLA, LCA, MV, WV) con aplicación de algoritmo DFP por cada iteración
    prediccionesOLApruned = ola(Cpruned, X_train, y_train, X_test, y_test, k)
    prediccionesLCApruned = lca(Cpruned, X_train, y_train, X_test, y_test, k)
    prediccionesMVpruned = majorityVoting(Cpruned, X_test)
    prediccionesWVpruned = weigthVoting(Cpruned, n_clases, X_test)

    flagMetric = True # Si flagMetric == False score Accuracy, True score AUC

    if  flagMetric == False:# Metrics accuracy

        # Obtenemos el valor exactitud de cada tecnica de selección dinámica (OLA, LCA, MV, WV) por cada iteración
        accuracyOLA.append(metrics.accuracy_score(y_test, prediccionesOLA) * 100.0)
        accuracyLCA.append(metrics.accuracy_score(y_test, prediccionesLCA) * 100.0)
        accuracyMV.append(metrics.accuracy_score(y_test, prediccionesMV) * 100.0)
        accuracyWV.append(metrics.accuracy_score(y_test, prediccionesWV) * 100.0)

        # Obtenemos el valor exactitud de cada tecnica de selección dinámica (OLA, LCA, MV, WV) con aplicación de algoritmo DFP por cada iteración
        accuracyOLApruned.append(metrics.accuracy_score(y_test, prediccionesOLApruned) * 100.0)
        accuracyLCApruned.append(metrics.accuracy_score(y_test, prediccionesLCApruned) * 100.0)
        accuracyMVpruned.append(metrics.accuracy_score(y_test, prediccionesMVpruned) * 100.0)
        accuracyWVpruned.append(metrics.accuracy_score(y_test, prediccionesWVpruned) * 100.0)
    else: # Metrics AUC

        # El AUC en un punto es lo mismo que sacar la media del porcentaje de acierto para cada clase.
        # Para ello, se obtiene la matriz de confusión, calcular el porcentaje de acierto de cada una de las clases
        # y calcular la media aritmética entre los porcentajes.

        # Obtenemos el valor AUC de cada tecnica de selección dinámica (OLA, LCA, MV, WV) por cada iteración
        cm_OLA = metrics.confusion_matrix(y_test, prediccionesOLA)
        cm_LCA = metrics.confusion_matrix(y_test, prediccionesLCA)
        cm_MV = metrics.confusion_matrix(y_test, prediccionesMV)
        cm_WV = metrics.confusion_matrix(y_test, prediccionesWV)

        accuracyOLA.append(media_aritmetica(cm_OLA) * 100)
        accuracyLCA.append(media_aritmetica(cm_LCA) * 100)
        accuracyMV.append(media_aritmetica(cm_MV) * 100)
        accuracyWV.append(media_aritmetica(cm_WV) * 100)

        # Obtenemos el valor AUC de cada tecnica de selección dinámica (OLA, LCA, MV, WV) con aplicación de algoritmo DFP por cada iteración
        cm_OLA_pruned = metrics.confusion_matrix(y_test, prediccionesOLApruned)
        cm_LCA_pruned = metrics.confusion_matrix(y_test, prediccionesLCApruned)
        cm_MV_pruned = metrics.confusion_matrix(y_test, prediccionesMVpruned)
        cm_WV_pruned = metrics.confusion_matrix(y_test, prediccionesWVpruned)

        accuracyOLApruned.append(media_aritmetica(cm_OLA_pruned) * 100)
        accuracyLCApruned.append(media_aritmetica(cm_LCA_pruned) * 100)
        accuracyMVpruned.append(media_aritmetica(cm_MV_pruned) * 100)
        accuracyWVpruned.append(media_aritmetica(cm_WV_pruned) * 100)

#Calculamos la media  del valor exactitud de cada tecnica de selección dinámica (OLA, LCA, MV, WV)
print "Para OLA  % de acierto: " + str(np.round(np.mean(accuracyOLA),2))
print "Para LCA % de acierto: " + str(np.round(np.mean(accuracyLCA),2))
print "Para MV  % de acierto: " + str(np.round(np.mean(accuracyMV),2))
print "Para WV  % de acierto: " + str(np.round(np.mean(accuracyWV),2))

#Calculamos la media  del valor exactitud de cada tecnica de selección dinámica (OLA, LCA, MV, WV) con aplicación de algoritmo DFP
print "Para OLA_con_DFP % de acierto: " + str(np.round(np.mean(accuracyOLApruned),2))
print "Para LCA_con_DFP % de acierto: " + str(np.round(np.mean(accuracyLCApruned),2))
print "Para MV_con_DFP  % de acierto: " + str(np.round(np.mean(accuracyMVpruned),2))
print "Para WV_con_DFP  % de acierto: " + str(np.round(np.mean(accuracyWVpruned),2))

tiempo_final = time()

tiempo_ejecucion = np.round_(tiempo_final - tiempo_inicial,1)
hora=(int(tiempo_ejecucion/3600))
min=int((tiempo_ejecucion-(hora*3600))/60)
seg=np.round_(tiempo_ejecucion-((hora*3600)+(min*60)),1)
print '\nEl tiempo de ejecucion fue: '+str(hora)+" horas "+str(min)+" minutos "+str(seg)+" segundos"
