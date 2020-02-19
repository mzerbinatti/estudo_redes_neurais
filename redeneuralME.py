import numpy as np
from abc import ABC, abstractmethod
import camadaRN as Camada

class RedeNeural_MLP:
    def __init__(self):
        self._tipo_rede_neural = 'batch' # padrão é BATCH

    @staticmethod
    def criar_rede_neural(tipo_rede_neural='batch', dados=None):
        

        if tipo_rede_neural == 'batch':
            return RedeNeural_MLP_Batch(dados)
        elif tipo_rede_neural == 'estocastico':
            return RedeNeural_MLP_Estocastico(dados)
        elif tipo_rede_neural == 'recorrente':
            return RedeNeural_Recorrente(dados)
        else:
            return RedeNeural_MLP_ME(dados)


class RedeNeuralAbstrata_MLP(ABC):

    def __init__(self, dados):
        self._camadas = []
        self.dados = dados

    # def adicionar_camada(self, qtd_entradas, qtde_neuronios, tipo_funcao_ativacao=None):
    #     self.adicionar_camada(Camada(qtd_entradas, qtde_neuronios, tipo_funcao_ativacao)) 

    def adicionar_camada(self, camada):
        self._camadas.append(camada)

    def feed_forward(self, X):
        for camada in self._camadas:
            X = camada.funcao_soma(X)
        self.valor_saida = X
        return X

    @abstractmethod
    def backpropagation(self, X, y, taxa_aprendizagem):
        pass

    @abstractmethod
    def treinar(self, X, Y, taxa_aprendizagem, maximo_epocas):
        pass

    def predizer(self, X, Y):
        valor_saida = self.feed_forward(X)
        mse = self.calcular_erro_medio_quadratico(valor_saida, Y)
        return valor_saida, mse
    
    def calcular_erro_medio_quadratico(self, valor_saida, Y, imprimir=False):
        mse = np.mean(np.square(Y - valor_saida))
        if (imprimir):
            print('MSE: %f' % (float(mse)))
        return mse

    # def calcular_erro_medio_quadratico(self, X, Y, imprimir=False):
    #     # mse = np.mean(np.square(Y - self.feed_forward(X)))
    #     saida = self.feed_forward(X)
    #     mse = (1/len(saida)) * np.sum((Y - saida) ** 2)
    #     if (imprimir):
    #         print('MSE: %f' % (float(mse)))
    #     return mse

class RedeNeural_MLP_Batch(RedeNeuralAbstrata_MLP):

    def backpropagation(self, X, y, taxa_aprendizagem):
        # Saida do Feed forward
        valor_saida = self.feed_forward(X)

        # ### TESTE MICHEL
        # valor_saida = self.dados.desnormalizar(valor_saida)
        # y = self.dados.desnormalizar(y) 
        # ### TESTE MICHEL

        # Loop nas camadas anteriores, partindo da de saida para a primeira
        for i in reversed(range(len(self._camadas))):
            camada = self._camadas[i]

            # verifica se é a camada de saída
            if camada == self._camadas[-1]:
                # camada.error = y - valor_saida
                camada.error = valor_saida - y
                # A saída é igual a camada.valor_ultima_funcao_ativacao neste caso
                camada.delta = camada.error * camada.funcao_ativacao_derivada(valor_saida)
            else:
                proxima_camada = self._camadas[i + 1]
                camada.error = np.dot(proxima_camada.delta, proxima_camada.pesos.T) # Alterado AQUI
                camada.delta = camada.error * camada.funcao_ativacao_derivada(camada.valor_ultima_funcao_ativacao)

        # Atualiza os pesos
        for i in range(len(self._camadas)):
            camada = self._camadas[i]
            #  A entrada é ou a saída da camada anterior ou o próprio X(para a primeira camada oculta)
            entrada_a_usar = np.atleast_2d(X if i == 0 else self._camadas[i - 1].valor_ultima_funcao_ativacao)
            camada.pesos -= np.dot(entrada_a_usar.T, camada.delta) * taxa_aprendizagem  # Alterado AQUI

    def treinar(self, taxa_aprendizagem, maximo_epocas):
        return self.treinar(self.dados.X_treino, self.dados.Y_treino, taxa_aprendizagem, maximo_epocas)

    def treinar(self, X, Y, taxa_aprendizagem, maximo_epocas):

        mses = []
        for i in range(maximo_epocas):
            self.backpropagation(X, Y, taxa_aprendizagem) # Alterado AQUI
            # if i % 10 == 0:
            valor_saida = self.feed_forward(X)

            # TESTE MICHEL            
            # valor_saida = self.dados.desnormalizar(valor_saida)
            # Y_desn = self.dados.desnormalizar(Y)
            # mse = np.mean(np.square(Y_desn - valor_saida))
            # TESTE MICHEL

            mse = np.mean(np.square(Y - valor_saida))
            mses.append(mse)
            # print('Época: #%s, MSE: %f' % (i, float(mse)))

        return mses, valor_saida

   
class RedeNeural_MLP_Estocastico(RedeNeuralAbstrata_MLP):

    def backpropagation(self, X, y, taxa_aprendizagem):
        # Saida do Feed forward
        valor_saida = self.feed_forward(X)

        # Loop nas camadas anteriores, partindo da de saida para a primeira
        for i in reversed(range(len(self._camadas))):
            camada = self._camadas[i]

            # verifica se é a camada de saída
            if camada == self._camadas[-1]:
                camada.error = y - valor_saida
                # A saída é igual a camada.valor_ultima_funcao_ativacao neste caso
                camada.delta = camada.error * camada.funcao_ativacao_derivada(valor_saida)
            else:
                proxima_camada = self._camadas[i + 1]
                camada.error = np.dot(proxima_camada.pesos, proxima_camada.delta)
                camada.delta = camada.error * camada.funcao_ativacao_derivada(camada.valor_ultima_funcao_ativacao)

        # Atualiza os pesos
        for i in range(len(self._camadas)):
            camada = self._camadas[i]
            #  A entrada é ou a saída da camada anterior ou o próprio X(para a primeira camada oculta)
            entrada_a_usar = np.atleast_2d(X if i == 0 else self._camadas[i - 1].valor_ultima_funcao_ativacao)
            camada.pesos += camada.delta * entrada_a_usar.T * taxa_aprendizagem
    
    def treinar(self, taxa_aprendizagem, maximo_epocas):
        return self.treinar(self.dados.X_treino, self.dados.Y_treino, taxa_aprendizagem, maximo_epocas)

    def treinar(self, X, Y, taxa_aprendizagem, maximo_epocas):

        mses = []

        for i in range(maximo_epocas):
            for j in range(len(X)):
                self.backpropagation(X[j], Y[j], taxa_aprendizagem)
            # if i % 10 == 0:
            valor_saida = self.feed_forward(X)
            mse = np.mean(np.square(Y - valor_saida))
            mses.append(mse)
            # print('Época: #%s, MSE: %f' % (i, float(mse)))

        return mses, valor_saida




class RedeNeural_MLP_ME(RedeNeuralAbstrata_MLP):

    def backpropagation(self, X, y, taxa_aprendizagem, Yg = 1):
        # Saida do Feed forward
        # valor_saida = self.feed_forward(X)

        # Loop nas camadas anteriores, partindo da de saida para a primeira
        for i in reversed(range(len(self._camadas))):
            camada = self._camadas[i]

            # verifica se é a camada de saída
            if camada == self._camadas[-1]:
                camada.error = Yg * (self.valor_saida - y)
                # A saída é igual a camada.valor_ultima_funcao_ativacao neste caso
                camada.delta = camada.error * camada.funcao_ativacao_derivada(self.valor_saida)
            else:
                proxima_camada = self._camadas[i + 1]
                camada.error = np.dot(proxima_camada.delta, proxima_camada.pesos.T) # Alterado AQUI
                camada.delta = camada.error * camada.funcao_ativacao_derivada(camada.valor_ultima_funcao_ativacao)

        # Atualiza os pesos
        for i in range(len(self._camadas)):
            camada = self._camadas[i]
            #  A entrada é ou a saída da camada anterior ou o próprio X(para a primeira camada oculta)
            entrada_a_usar = np.atleast_2d(X if i == 0 else self._camadas[i - 1].valor_ultima_funcao_ativacao)
            gradiente = np.dot(entrada_a_usar.T, camada.delta) / entrada_a_usar.shape[0] # VERIFICAR AQUI
            camada.pesos -= taxa_aprendizagem * gradiente   # Verificar se esse passo existe

    # def calcular_gradiente_especialista_RN(sinal, a2, a3, pesos_camada_2, pesos_camada_3_saida, data):
    #     a3_delta = sinal * a3 * (1 - a3)
    #     pesos_gradiente_camada_3_saida = a2.T.dot(a3_delta) / data.shape[0]
    #     a2_grad = a3_delta.dot(pesos_camada_3_saida[1:, :].T)
    #     a2_delta = a2_grad * a2[:, 1:] * (1 - a2[:, 1:])
    #     pesos_gradiente_camada_2 = data.T.dot(a2_delta) / data.shape[0]
    #     return pesos_gradiente_camada_2, pesos_gradiente_camada_3_saida

    def treinar(self, taxa_aprendizagem, maximo_epocas):
        return self.treinar(dados.X_treino, dados.Y_treino, taxa_aprendizagem, maximo_epocas)

    def treinar(self, X, Y, taxa_aprendizagem, maximo_epocas):

        mses = []
        for i in range(maximo_epocas):
            self.backpropagation(X, Y, taxa_aprendizagem) # Alterado AQUI
            if i % 10 == 0:
                mse = np.mean(np.square(Y - self.feed_forward(X)))
                mses.append(mse)
                print('Época: #%s, MSE: %f' % (i, float(mse)))

        return mses
    
    def feed_forward(self, X):
        for camada in self._camadas:
            X = camada.funcao_soma(X)
        self.valor_saida = X
        return X


class RedeNeural_Recorrente(RedeNeuralAbstrata_MLP):

    def __init__(self, dados):
        super().__init__(dados)
        self.dados = dados
        # self.camada_peso_recorrencia = []
    
    def adicionar_camada(self, camada):
        super().adicionar_camada(camada)
        # self.camada_peso_recorrencia.append(Camada.CamadaRecorrente(1, camada.n_neuronios, camada.tipo_funcao_ativacao))

    def feed_forward(self, X, treinamento=False):
        # for camada in self._camadas:
        for camada in self._camadas:
            X = camada.funcao_soma(X, treinamento)

        self.valor_saida = X
        return X

    def backpropagation(self, X, y, taxa_aprendizagem):
        # Saida do Feed forward
        for j in range(len(X)):
            valor_saida = self.feed_forward(X[j], True)

        valor_saida  = self.feed_forward(X, False)

        # Loop nas camadas anteriores, partindo da de saida para a primeira
        for i in reversed(range(len(self._camadas))):
            camada = self._camadas[i]

            # verifica se é a camada de saída
            if camada == self._camadas[-1]:
                # camada.error = y - valor_saida
                camada.error = (y - valor_saida)
                # camada.error = -(y*np.log(valor_saida) + (1-y)*np.log(1-valor_saida))
                # A saída é igual a camada.valor_ultima_funcao_ativacao neste caso
                camada.delta = camada.error * camada.funcao_ativacao_derivada(valor_saida)
            else:
                proxima_camada = self._camadas[i + 1]
                camada.error = np.dot(proxima_camada.delta, proxima_camada.pesos.T)
                camada.delta = camada.error * camada.funcao_ativacao_derivada(camada.valor_ultima_funcao_ativacao)

        # Atualiza os pesos
        for i in range(len(self._camadas)):
            camada = self._camadas[i]
            #  A entrada é ou a saída da camada anterior ou o próprio X(para a primeira camada oculta)
            entrada_a_usar = np.atleast_2d(X if i == 0 else self._camadas[i - 1].valor_ultima_funcao_ativacao)
            entrada_a_usar_rec = np.atleast_2d(X if i == 0 else self._camadas[i].valor_recorrente)
            camada.pesos += np.dot(entrada_a_usar.T, camada.delta) * taxa_aprendizagem
            # camada.camada_recorrente.pesos += np.dot(entrada_a_usar_rec.T, camada.delta) * taxa_aprendizagem
            
    def treinar(self, taxa_aprendizagem, maximo_epocas):
        return self.treinar(dados.X_treino, dados.Y_treino, taxa_aprendizagem, maximo_epocas)

    def treinar(self, X, Y, taxa_aprendizagem, maximo_epocas):
        mses = []
        for i in range(maximo_epocas):
            # for j in range(len(X)):
                # self.backpropagation(X[j], Y[j], taxa_aprendizagem)
            self.backpropagation(X, Y, taxa_aprendizagem)
            # if i % 10 == 0:
            # if i == 4999:
            #     a = 1
            
            valor_saida = self.feed_forward(X)
            # mse = np.sum(-(Y*np.log(valor_saida)+(1-Y)*np.log(1-valor_saida)))
            # mse = np.mean(np.square(Y - valor_saida))
            mse = self.calcular_erro_medio_quadratico(valor_saida, Y)
            mses.append(mse)
            # print('Época: #%s, MSE: %f' % (i, float(mse)))

        # return mses, valor_saida, Y
        return mses, valor_saida