import numpy as np

class Dados:

    def __init__(self, X, lags=0, tipo_normalizacao=None, percentual_treino=0.7, minimo_normalizacao=0, maximo_normalizacao=1):
        self.tipo_normalizacao = tipo_normalizacao
        self.percentual_treino=percentual_treino
        self.minimo_normalizacao = minimo_normalizacao
        self.maximo_normalizacao = maximo_normalizacao
        self.lags = lags
        # self.X, self.Y = self.preparar_array_X_Y(X, lags)
        self.X = X
        self.X_original = X

    def preparar_dados(self):
        self.X_dados_normalizados = self.normalizar()
        self.X, self.Y = self.preparar_array_X_Y(self.X, self.lags)
        self.Y = np.reshape(self.Y, (len(self.Y), 1))
        self.X_dados_normalizados, self.Y_dados_normalizados = self.preparar_array_X_Y(self.X_dados_normalizados, self.lags)
        self.Y_dados_normalizados = np.reshape(self.Y_dados_normalizados, (len(self.Y_dados_normalizados), 1))

        qtd_treino = int(len(self.X_dados_normalizados) * self.percentual_treino)
        
        self.X_treino = self.X_dados_normalizados[0:qtd_treino]
        self.Y_treino = self.Y_dados_normalizados[0:qtd_treino]

        self.X_validacao = self.X_dados_normalizados[qtd_treino:]
        self.Y_validacao = self.Y_dados_normalizados[qtd_treino:]

    def normalizar(self):
        if self.tipo_normalizacao=='minmax':
            self.X_dados_normalizados = self.normalizar_dados_minmax()
        elif self.tipo_normalizacao=='zscore':
            self.X_dados_normalizados = self.normalizar_dados_zscore()
        else: # Não normaliza
            self.X_dados_normalizados = self.X

        return self.X_dados_normalizados

    def desnormalizar(self, X_norm):
        if self.tipo_normalizacao=='minmax':
            return self.desnormalizar_dados_minmax(X_norm)
        elif self.tipo_normalizacao=='zscore':
            return self.desnormalizar_dados_zscore(X_norm)
        else: # Não normaliza
            return X_norm

    def normalizar_dados_minmax(self):
        self.maximo_serie = np.amax(self.X)
        self.minimo_serie = np.amin(self.X)
        self.X_normalizado  = self.minimo_normalizacao + ((self.X - self.minimo_serie) / (self.maximo_serie - self.minimo_serie)) * (self.maximo_normalizacao - self.minimo_normalizacao)
        return self.X_normalizado

    def desnormalizar_dados_minmax(self, X_norm):
        X_desnormalizado = X_norm * (self.maximo_serie - self.minimo_serie) + self.minimo_serie
        return X_desnormalizado

    # Standardize - Desvio padrao
    def normalizar_dados_zscore(self): 
        tamanho = len(self.X)
        soma = np.sum(self.X)
        self.media_serie = soma / tamanho
        self.desvio_padrao = np.sqrt( np.sum( (self.X - self.media_serie)**2 ) / tamanho )
        self.X_normalizado = (self.X - self.media_serie) / self.desvio_padrao
        return self.X_normalizado
    
    # Standardize - Desvio padrao
    def desnormalizar_dados_zscore(self, X_norm): 
        X_desnormalizado = (X_norm * self.desvio_padrao) + self.media_serie
        return X_desnormalizado


    # Prepara array X e Y de acordo com o LAG
    def preparar_array_X_Y(self, serie, lag):
        serie_X = np.zeros(((len(serie) - lag), lag))
        serie_Y = np.zeros((len(serie) - lag))
        
        for index_X in range(lag, len(serie)):
            lag_conta = lag
            for item_X in range(lag): #percorre de 0 até lag (tamanho do histórico considerado)
                serie_X[index_X - lag][item_X] = serie[index_X - lag_conta]
                lag_conta = lag_conta - 1
            serie_Y[index_X - lag] = serie[index_X]
        
        return serie_X, serie_Y