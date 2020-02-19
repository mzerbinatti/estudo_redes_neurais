import numpy as np

class ManipulacaoDados:

    @staticmethod
    def carregar_arquivo(caminho_arquivo):
        content_array = []
        import os
        pasta = os.path.dirname(os.path.abspath(__file__))
        endereco_arquivo = os.path.join(pasta, caminho_arquivo)

        with open(endereco_arquivo) as f:
                for line in f:
                    content_array.append(float(line.replace('\n', '')))
                print(content_array)
        return content_array  
    
    @staticmethod
    def preparar_array_X_Y(serie, lag):
        serie_X = np.zeros(((len(serie) - lag), lag))
        serie_Y = np.zeros((len(serie) - lag))
        
        for index_X in range(lag, len(serie)):
            lag_conta = lag
            for item_X in range(lag): #percorre de 0 até lag (tamanho do histórico considerado)
                serie_X[index_X - lag][item_X] = serie[index_X - lag_conta]
                lag_conta = lag_conta - 1
            serie_Y[index_X - lag] = serie[index_X]
        
        return serie_X, serie_Y
    
    @staticmethod
    def normalizar_dados(serie, minimo_esperado=0, maximo_esperado=1):
        maximo = np.amax(serie)
        minimo = np.amin(serie)
        nova_serie = minimo_esperado + ((serie - minimo) / (maximo - minimo)) * (maximo_esperado - minimo_esperado)
        return nova_serie
    
    @staticmethod
    def desnormalizar(valor):
        valor_desnormalizado = valor * self.desvio_padrao + self.media
        return valor_desnormalizado


    @staticmethod
    def calcular_lag(serie):
        media = np.mean(serie)
        tamanho_serie = len(serie)
        maximo_lag_calculado = 20
        i = 0
        j = 0
        array_autocorrelacao = np.zeros((maximo_lag_calculado + 1, 1))
        for i in range(maximo_lag_calculado + 1):
            numerador = 0
            denominador = 0
            j = 0
            while j <  (tamanho_serie - maximo_lag_calculado):
                numerador = numerador + (serie[j] - media)*(serie[j+i] - media)
                denominador = denominador + (serie[j] - media) ** 2
                j = j + 1
            array_autocorrelacao[i] = numerador / denominador

        return array_autocorrelacao

    @staticmethod
    def standardize_dados(serie): 
        tamanho = len(serie)
        soma = np.sum(serie)
        media = soma / tamanho
        desvio_padrao = np.sqrt( np.sum( (serie - media)**2 ) / tamanho )
        nova_serie = (serie - media) / desvio_padrao
        return nova_serie