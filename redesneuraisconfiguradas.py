import numpy as np
import matplotlib.pyplot as plt
import redeneuralME as rn_mlp_ME
import camadaRN as camada
# import mistura as me
import auxiliar as manipulacao
import dados as dados_serie
import time
import mistura_michel as mistura

import plotly.graph_objs as go

# np.random.seed(10)

# max_epocas = 10000
# taxa_aprendizagem = 0.01

# serie1_txt = np.array(manipulacao.ManipulacaoDados.carregar_arquivo('dados/serie1_trein.txt'))
# X, Y = manipulacao.ManipulacaoDados.preparar_array_X_Y(serie1_txt, 5)
# Y = np.reshape(Y, (len(Y), 1))

###########################################################################################################################
## 1º Experimento - REDE MLP - BATCH
###########################################################################################################################


class RedesNeuraisConfiguradas:

    # def __init__(self, X_treino, Y_treino, X_validacao, Y_validacao, taxa_aprendizagem=0.01, max_epocas=5000):
    def __init__(self, dados, taxa_aprendizagem=0.01, max_epocas=5000):
        # self.X_treino = X_treino
        # self.Y_treino = Y_treino
        # self.X_validacao = X_validacao
        # self.Y_validacao = Y_validacao
        self.dados = dados
        self.dados.preparar_dados()
        self.taxa_aprendizagem = taxa_aprendizagem
        self.max_epocas = max_epocas
        self.qtd_epocas = np.array(list(range(max_epocas))) + 1
        self.qtd_treino  = np.array(list(range(len(self.dados.X_treino)))) + 1
        self.qtd_validacao = np.array(list(range(len(self.dados.X_validacao)))) + 1

    
    def treinar_rede(self, nn_MLP):
        print("Rede: ", self.nome_rede)

        # Erro INICIAL - PRÉ-TREINAMENTO
        _, erro_medio_pre_MLP = nn_MLP.predizer(self.dados.X_treino, self.dados.Y_treino)
        print("EQM Pré-Treinamemnto: ", erro_medio_pre_MLP)

        # Treinamento
        inicio = time.time()
        erros_medios_MLP, saida_treinamento_MLP = nn_MLP.treinar(self.dados.X_treino, self.dados.Y_treino, self.taxa_aprendizagem, self.max_epocas)
        fim = time.time()

        # Tempo Processamento
        tempo_MLP = fim - inicio

        # Erro FINAL
        _, erro_medio_pos_MLP = nn_MLP.predizer(self.dados.X_treino, self.dados.Y_treino)
        print("EQM Pós-Treinamento: ", erro_medio_pos_MLP)

        
        # Saida e Erro dados Validacao
        predicao_MLP, mse_validacao_MLP = nn_MLP.predizer(self.dados.X_validacao, self.dados.Y_validacao)
        print("EQM Validação: ", mse_validacao_MLP)
        
        # Desnormalizando os dados
        predicao_MLP = self.dados.desnormalizar(predicao_MLP)
        saida_treinamento_MLP = self.dados.desnormalizar(saida_treinamento_MLP)

        # Dados Plot Erro Médio Quadratico
        trace_erros_medios = go.Scatter(
            x = self.qtd_epocas,
            y = erros_medios_MLP,
            name= self.nome_rede
        )

        # Dados Plot Erro Médio Quadratico
        trace_Treino = go.Scatter(
            x = self.qtd_treino,
            y = np.array(saida_treinamento_MLP).reshape(-1),
            name = self.nome_rede
        )

        # Dados Plot Erro Médio Quadratico
        trace_validacao = go.Scatter(
            x = self.qtd_validacao,
            y = predicao_MLP.reshape(-1),
            name = self.nome_rede
        )

        print("Tempo de execução: ", tempo_MLP)

        return erros_medios_MLP, erro_medio_pre_MLP, erro_medio_pos_MLP, saida_treinamento_MLP, tempo_MLP, predicao_MLP, mse_validacao_MLP, trace_erros_medios, trace_Treino, trace_validacao

    
    def treinar_rede_especialista(self, nn_E):
        print("Rede: ", self.nome_rede)

        # Erro INICIAL - PRÉ-TREINAMENTO
        erro_medio_pre_ME = nn_E.calcular_erro_quadratico_medio_total_ME(self.dados.X_treino, self.dados.Y_treino, True)
        print("EQM Pré-Treinamemnto: ", erro_medio_pre_ME)

        # Treinamento
        inicio = time.time()
        erros_medios_ME, erro_ESPECIALISTA_ME, saida_treinamento_ME = nn_E.treinar(self.dados.X_treino, self.dados.Y_treino, self.max_epocas, self.taxa_aprendizagem)
        fim = time.time()

        # Tempo Processamento
        tempo_ME = fim - inicio

        # Erro FINAL
        erro_medio_pos_ME = nn_E.calcular_erro_quadratico_medio_total_ME(self.dados.X_treino, self.dados.Y_treino, True)
        print("EQM Pós-Treinamemnto: ", erro_medio_pos_ME)

        # Saida e Erro dados Validacao
        predicao_ME, mse_validacao_ME = nn_E.predizer(self.dados.X_validacao, self.dados.Y_validacao)
        print("EQM Validação: ", mse_validacao_ME)

        # Desnormalizando os dados
        predicao_ME = self.dados.desnormalizar(predicao_ME)
        saida_treinamento_ME = self.dados.desnormalizar(saida_treinamento_ME)

        # Dados Plot Erro Médio Quadratico
        trace_erros_medios = go.Scatter(
            x = self.qtd_epocas,
            y = erros_medios_ME,
            name= self.nome_rede
        )

        # Dados Plot Erro Médio Quadratico
        trace_Treino = go.Scatter(
            x = self.qtd_treino,
            y = np.array(saida_treinamento_ME).reshape(-1),
            name = self.nome_rede
        )

        # Dados Plot Erro Médio Quadratico
        trace_validacao = go.Scatter(
            x = self.qtd_validacao,
            y = np.array(predicao_ME).reshape(-1),
            name = self.nome_rede
        )

        print("Tempo de execução: ", tempo_ME)

        return erros_medios_ME, erro_medio_pre_ME, erro_medio_pos_ME, saida_treinamento_ME, tempo_ME, predicao_ME, mse_validacao_ME, trace_erros_medios, trace_Treino, trace_validacao


    def mlp_batch_1(self):
        self.nome_rede = 'MLP Batch 1'

        MLP_B2 = '''
        ##########################
        ## MLP 2 camadas
        ## Camada 1:        5 Neurônios  - Sigmoide
        ## Camada Saída:    1 Neurônio   - Tangente
        ##########################
        '''

        print(MLP_B2 + "\n")
        nn_MLP_B2 = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('batch', self.dados)

        # Definir camadas da Rede
        nn_MLP_B2.adicionar_camada(camada.Camada(len(self.dados.X_treino[0]), 5, 'sigmoid')) 
        nn_MLP_B2.adicionar_camada(camada.Camada(5, 1, 'tanh')) 

        erros_medios_MLP_B2, erro_medio_pre_MLP_B2, erro_medio_pos_MLP_B2, saida_treinamento_MLP_B2, tempo_MLP_B2, predicao_MLP_B2, mse_validacao_MLP_B2, trace_erros_medios, trace_Treino, trace_validacao = self.treinar_rede(nn_MLP_B2)

        return erros_medios_MLP_B2, erro_medio_pre_MLP_B2, erro_medio_pos_MLP_B2, saida_treinamento_MLP_B2, tempo_MLP_B2, predicao_MLP_B2, mse_validacao_MLP_B2, trace_erros_medios, trace_Treino, trace_validacao
        ##############################################################################


    def mlp_batch_2(self):
        self.nome_rede = 'MLP Batch 2'
        MLP_B1 = '''
        ##########################
        ## MLP 3 camadas
        ## Camada 1:        3 Neurônios  - Tangente
        ## Camada 2:        10 Neurônios - Sigmoide
        ## Camada Saída:    1 Neurônio   - Sigmoide
        ##########################
        '''

        print(MLP_B1 + "\n")
        nn_MLP_B1 = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('batch', self.dados)

        # Definir camadas da Rede
        nn_MLP_B1.adicionar_camada(camada.Camada(len(self.dados.X_treino[0]), 3, 'tanh')) 
        nn_MLP_B1.adicionar_camada(camada.Camada(3, 10, 'sigmoid')) 
        nn_MLP_B1.adicionar_camada(camada.Camada(10, 1, 'sigmoid')) 


        # Treina a Rede
        erros_medios_MLP_B1, erro_medio_pre_MLP_B1, erro_medio_pos_MLP_B1, saida_treinamento_MLP_B1, tempo_MLP_B1, predicao_MLP_B1, mse_validacao_MLP_B1, trace_erros_medios_MLP_B1, trace_Treino_MLP_B1, trace_validacao_MLP_B1 = self.treinar_rede(nn_MLP_B1)

        return erros_medios_MLP_B1, erro_medio_pre_MLP_B1, erro_medio_pos_MLP_B1, saida_treinamento_MLP_B1, tempo_MLP_B1, predicao_MLP_B1, mse_validacao_MLP_B1, trace_erros_medios_MLP_B1, trace_Treino_MLP_B1, trace_validacao_MLP_B1

        ##############################################################################



    def mlp_batch_3(self):
        self.nome_rede = 'MLP Batch 3'
        MLP_B3 = '''
        ##########################
        ## MLP 4 camadas
        ## Camada 2:        5 Neurônios  - Tangente
        ## Camada 10:       10 Neurônios - Tangente
        ## Camada 4:        4 Neurônios  - Tangente
        ## Camada Saída:    1 Neurônio   - Tangente
        ##########################
        '''

        print(MLP_B3 + "\n")
        nn_MLP_B3 = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('batch', self.dados)

        # Definir camadas da Rede
        nn_MLP_B3.adicionar_camada(camada.Camada(len(self.dados.X_treino[0]), 5, 'tanh')) 
        nn_MLP_B3.adicionar_camada(camada.Camada(5, 10, 'tanh'))      
        nn_MLP_B3.adicionar_camada(camada.Camada(10, 4, 'tanh'))      
        nn_MLP_B3.adicionar_camada(camada.Camada(4, 1, 'tanh'))  
      

        # Treinamento
        erros_medios_MLP_B3, erro_medio_pre_MLP_B3, erro_medio_pos_MLP_B3, saida_treinamento_MLP_B3, tempo_MLP_B3, predicao_MLP_B3, mse_validacao_MLP_B3, trace_erros_medios, trace_Treino, trace_validacao = self.treinar_rede(nn_MLP_B3)

        return erros_medios_MLP_B3, erro_medio_pre_MLP_B3, erro_medio_pos_MLP_B3, saida_treinamento_MLP_B3, tempo_MLP_B3, predicao_MLP_B3, mse_validacao_MLP_B3, trace_erros_medios, trace_Treino, trace_validacao
        ##############################################################################


   
    ###########################################################################################################################
    ## Experimento - REDES MLP - ESTOCASTICAS
    ########################################################################################################################### 
    
    def mlp_estocastico_1(self):
        self.nome_rede = 'MLP Estocástica 1'
        MLP_E2 = '''
        ##########################
        ## MLP 2 camadas - Estocastico
        ## Camada 1:        5 Neurônios  - Sigmoide
        ## Camada Saída:    1 Neurônio   - Tangente
        ##########################
        '''

        print(MLP_E2 + "\n")
        nn_MLP_E2 = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('estocastico', self.dados)

        # Definir camadas da Rede
        nn_MLP_E2.adicionar_camada(camada.Camada(len(self.dados.X_treino[0]), 5, 'sigmoid')) # Linear
        nn_MLP_E2.adicionar_camada(camada.Camada(5, 1, 'tanh')) # Sigmoid

        # Treinamento
        erros_medios_MLP_E2, erro_medio_pre_MLP_E2, erro_medio_pos_MLP_E2, saida_treinamento_MLP_E2, tempo_MLP_E2, predicao_MLP_E2, mse_validacao_MLP_E2, trace_erros_medios, trace_Treino, trace_validacao = self.treinar_rede(nn_MLP_E2)

        return erros_medios_MLP_E2, erro_medio_pre_MLP_E2, erro_medio_pos_MLP_E2, saida_treinamento_MLP_E2, tempo_MLP_E2, predicao_MLP_E2, mse_validacao_MLP_E2, trace_erros_medios, trace_Treino, trace_validacao
        ##############################################################################

    def mlp_estocastico_2(self):
        self.nome_rede = 'MLP Estocástica 2'
        MLP_E1 = '''
        ##########################
        ## MLP 3 camadas - Estocastico
        ## Camada 1:        5 Neurônios  - Sigmoide
        ## Camada 2:        10 Neurônios - Sigmoide
        ## Camada Saída:    1 Neurônio   - Tangente
        ##########################
        '''

        print(MLP_E1 + "\n")
        nn_MLP_E1 = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('estocastico', self.dados)

        # Definir camadas da Rede
        nn_MLP_E1.adicionar_camada(camada.Camada(len(self.dados.X_treino[0]), 3, 'tanh')) 
        nn_MLP_E1.adicionar_camada(camada.Camada(3, 10, 'sigmoid')) 
        nn_MLP_E1.adicionar_camada(camada.Camada(10, 1, 'sigmoid')) 



        # Treinamento
        erros_medios_MLP_E1, erro_medio_pre_MLP_E1, erro_medio_pos_MLP_E1, saida_treinamento_MLP_E1, tempo_MLP_E1, predicao_MLP_E1, mse_validacao_MLP_E1, trace_erros_medios, trace_Treino, trace_validacao = self.treinar_rede(nn_MLP_E1)

        return erros_medios_MLP_E1, erro_medio_pre_MLP_E1, erro_medio_pos_MLP_E1, saida_treinamento_MLP_E1, tempo_MLP_E1, predicao_MLP_E1, mse_validacao_MLP_E1, trace_erros_medios, trace_Treino, trace_validacao
        ##############################################################################



    def mlp_estocastico_3(self):
        self.nome_rede = 'MLP Estocástica 3'
        MLP_E3 = '''
        ##########################
        ## MLP 4 camadas - Estocastico
        ## Camada 2:        2 Neurônios  - Tangente
        ## Camada 10:       10 Neurônios - Tangente
        ## Camada 4:        4 Neurônios  - Tangente
        ## Camada Saída:    1 Neurônio   - Tangente
        ##########################
        '''

        print(MLP_E3 + "\n")
        nn_MLP_E3 = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('estocastico', self.dados)

        # Definir camadas da Rede
        nn_MLP_E3.adicionar_camada(camada.Camada(len(self.dados.X_treino[0]), 2, 'tanh')) 
        nn_MLP_E3.adicionar_camada(camada.Camada(2, 10, 'tanh'))      
        nn_MLP_E3.adicionar_camada(camada.Camada(10, 4, 'tanh'))      
        nn_MLP_E3.adicionar_camada(camada.Camada(4, 1, 'tanh'))     

        # Treinamento
        erros_medios_MLP_E3, erro_medio_pre_MLP_E3, erro_medio_pos_MLP_E3, saida_treinamento_MLP_E3, tempo_MLP_E3, predicao_MLP_E3, mse_validacao_MLP_E3, trace_erros_medios, trace_Treino, trace_validacao = self.treinar_rede(nn_MLP_E3)

        return erros_medios_MLP_E3, erro_medio_pre_MLP_E3, erro_medio_pos_MLP_E3, saida_treinamento_MLP_E3, tempo_MLP_E3, predicao_MLP_E3, mse_validacao_MLP_E3, trace_erros_medios, trace_Treino, trace_validacao

###########################################################################################################################
###########################################################################################################################



################################ 
## MISTURA ESPECIALISTA
################################

    def mistura_especialista_1(self):
        self.nome_rede = 'Mistura Especialista 1'
        ME_B2 = '''
        ##########################
        ## MLP 2 camadas - Mistura Especialista
        ## Camada 1:        5 Neurônios  - Sigmoide
        ## Camada Saída:    1 Neurônio   - Tangente
        ##########################
        '''

        print(ME_B2 + "\n")

        lista_especialistas = []
        nn1 = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('ME', self.dados)
        nn1.adicionar_camada(camada.Camada(len(self.dados.X_treino[0]), 5, 'sigmoid')) # Linear
        nn1.adicionar_camada(camada.Camada(5, 1, 'tanh')) # Sigmoid
        lista_especialistas.append(nn1)

        nn2 = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('ME', self.dados)
        nn2.adicionar_camada(camada.Camada(len(self.dados.X_treino[0]), 5, 'sigmoid')) # Linear
        nn2.adicionar_camada(camada.Camada(5, 1, 'tanh')) # Sigmoid
        lista_especialistas.append(nn2)

        nn3 = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('ME', self.dados)
        nn3.adicionar_camada(camada.Camada(len(self.dados.X_treino[0]), 5, 'sigmoid')) # Linear
        nn3.adicionar_camada(camada.Camada(5, 1, 'tanh')) # Sigmoid
        lista_especialistas.append(nn3)

        nn4 = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('ME', self.dados)
        nn4.adicionar_camada(camada.Camada(len(self.dados.X_treino[0]), 5, 'sigmoid')) # Linear
        nn4.adicionar_camada(camada.Camada(5, 1, 'tanh')) # Sigmoid
        lista_especialistas.append(nn4)

        nn5 = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('ME', self.dados)
        nn5.adicionar_camada(camada.Camada(len(self.dados.X_treino[0]), 5, 'sigmoid')) # Linear
        nn5.adicionar_camada(camada.Camada(5, 1, 'tanh')) # Sigmoid
        lista_especialistas.append(nn5)

        nn6 = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('ME', self.dados)
        nn6.adicionar_camada(camada.Camada(len(self.dados.X_treino[0]), 5, 'sigmoid')) # Linear
        nn6.adicionar_camada(camada.Camada(5, 1, 'tanh')) # Sigmoid
        lista_especialistas.append(nn6)

        nn7 = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('ME', self.dados)
        nn7.adicionar_camada(camada.Camada(len(self.dados.X_treino[0]), 5, 'sigmoid')) # Linear
        nn7.adicionar_camada(camada.Camada(5, 1, 'tanh')) # Sigmoid
        lista_especialistas.append(nn7)

        nn8 = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('ME', self.dados)
        nn8.adicionar_camada(camada.Camada(len(self.dados.X_treino[0]), 5, 'sigmoid')) # Linear
        nn8.adicionar_camada(camada.Camada(5, 1, 'tanh')) # Sigmoid
        lista_especialistas.append(nn8)

        # ME2 = me.MisturaEspecialista(lista_especialistas)
        m = mistura.Mistura_Especialista(lista_especialistas)
        return m.mistura(self.dados)

        # Treinamento
        # erros_medios_ME2, erro_medio_pre_ME2, erro_medio_pos_ME2, saida_treinamento_ME2, tempo_ME2, predicao_ME2, mse_validacao_ME2, trace_erros_medios, trace_Treino, trace_validacao = self.treinar_rede_especialista(ME2)

        # return erros_medios_ME2, erro_medio_pre_ME2, erro_medio_pos_ME2, saida_treinamento_ME2, tempo_ME2, predicao_ME2, mse_validacao_ME2, trace_erros_medios, trace_Treino, trace_validacao




    def mistura_especialista_2(self):
        self.nome_rede = 'Mistura Especialista 2'
        ME_B1 = '''
        ##########################
        ## MLP 3 camadas - Mistura Especialista
        ## Camada 1:        5 Neurônios  - Sigmoide
        ## Camada 2:        10 Neurônios - Sigmoide
        ## Camada Saída:    1 Neurônio   - Tangente
        ##########################
        '''

        print(ME_B1 + "\n")
        lista_especialistas = []
        nn1 = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('ME', self.dados)
        nn1.adicionar_camada(camada.Camada(len(self.dados.X_treino[0]), 3, 'tanh')) 
        nn1.adicionar_camada(camada.Camada(3, 10, 'sigmoid')) 
        nn1.adicionar_camada(camada.Camada(10, 1, 'sigmoid')) 
        lista_especialistas.append(nn1)

        nn2 = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('ME', self.dados)
        nn2.adicionar_camada(camada.Camada(len(self.dados.X_treino[0]), 3, 'tanh')) 
        nn2.adicionar_camada(camada.Camada(3, 10, 'sigmoid')) 
        nn2.adicionar_camada(camada.Camada(10, 1, 'sigmoid')) 
        lista_especialistas.append(nn2)

        nn3 = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('ME', self.dados)
        nn3.adicionar_camada(camada.Camada(len(self.dados.X_treino[0]), 3, 'tanh')) 
        nn3.adicionar_camada(camada.Camada(3, 10, 'sigmoid')) 
        nn3.adicionar_camada(camada.Camada(10, 1, 'sigmoid')) 
        lista_especialistas.append(nn3)

        nn4 = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('ME', self.dados)
        nn4.adicionar_camada(camada.Camada(len(self.dados.X_treino[0]), 3, 'tanh')) 
        nn4.adicionar_camada(camada.Camada(3, 10, 'sigmoid')) 
        nn4.adicionar_camada(camada.Camada(10, 1, 'sigmoid')) 
        lista_especialistas.append(nn4)

        nn5 = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('ME', self.dados)
        nn5.adicionar_camada(camada.Camada(len(self.dados.X_treino[0]), 3, 'tanh')) 
        nn5.adicionar_camada(camada.Camada(3, 10, 'sigmoid')) 
        nn5.adicionar_camada(camada.Camada(10, 1, 'sigmoid')) 
        lista_especialistas.append(nn5)

        nn6 = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('ME', self.dados)
        nn6.adicionar_camada(camada.Camada(len(self.dados.X_treino[0]), 3, 'tanh')) 
        nn6.adicionar_camada(camada.Camada(3, 10, 'sigmoid')) 
        nn6.adicionar_camada(camada.Camada(10, 1, 'sigmoid')) 
        lista_especialistas.append(nn6)

        nn7 = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('ME', self.dados)
        nn7.adicionar_camada(camada.Camada(len(self.dados.X_treino[0]), 3, 'tanh')) 
        nn7.adicionar_camada(camada.Camada(3, 10, 'sigmoid')) 
        nn7.adicionar_camada(camada.Camada(10, 1, 'sigmoid')) 
        lista_especialistas.append(nn7)

        nn8 = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('ME', self.dados)
        nn8.adicionar_camada(camada.Camada(len(self.dados.X_treino[0]), 3, 'tanh')) 
        nn8.adicionar_camada(camada.Camada(3, 10, 'sigmoid')) 
        nn8.adicionar_camada(camada.Camada(10, 1, 'sigmoid')) 
        lista_especialistas.append(nn8)


        m = mistura.Mistura_Especialista(lista_especialistas)
        return m.mistura(self.dados)
        # ME1 = me.MisturaEspecialista(lista_especialistas)

        # # Treinamento
        # erros_medios_ME1, erro_medio_pre_ME1, erro_medio_pos_ME1, saida_treinamento_ME1, tempo_ME1, predicao_ME1, mse_validacao_ME1, trace_erros_medios, trace_Treino, trace_validacao = self.treinar_rede_especialista(ME1)

        # return erros_medios_ME1, erro_medio_pre_ME1, erro_medio_pos_ME1, saida_treinamento_ME1, tempo_ME1, predicao_ME1, mse_validacao_ME1, trace_erros_medios, trace_Treino, trace_validacao



    def mistura_especialista_3(self):
        self.nome_rede = 'Mistura Especialista 3'
        ME_B3 = '''
        ##########################
        ## MLP 4 camadas - Mistura Especialista
        ## Camada 2:        2 Neurônios  - Tangente
        ## Camada 10:       10 Neurônios - Sigmoide
        ## Camada 4:        4 Neurônios  - Tangente
        ## Camada Saída:    1 Neurônio   - Sigmoide
        ##########################
        '''


        print(ME_B3 + "\n")

        lista_especialistas = []
        nn1 = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('ME', self.dados)
        nn1.adicionar_camada(camada.Camada(len(self.dados.X_treino[0]), 2, 'tanh')) 
        nn1.adicionar_camada(camada.Camada(2, 10, 'tanh')) 
        nn1.adicionar_camada(camada.Camada(10, 4, 'tanh')) 
        nn1.adicionar_camada(camada.Camada(4, 1, 'tanh')) 
        lista_especialistas.append(nn1)

        nn2 = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('ME', self.dados)
        nn2.adicionar_camada(camada.Camada(len(self.dados.X_treino[0]), 2, 'tanh')) 
        nn2.adicionar_camada(camada.Camada(2, 10, 'tanh')) 
        nn2.adicionar_camada(camada.Camada(10, 4, 'tanh')) 
        nn2.adicionar_camada(camada.Camada(4, 1, 'tanh')) 
        lista_especialistas.append(nn2)

        nn3 = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('ME', self.dados)
        nn3.adicionar_camada(camada.Camada(len(self.dados.X_treino[0]), 2, 'tanh')) 
        nn3.adicionar_camada(camada.Camada(2, 10, 'tanh')) 
        nn3.adicionar_camada(camada.Camada(10, 4, 'tanh')) 
        nn3.adicionar_camada(camada.Camada(4, 1, 'tanh')) 
        lista_especialistas.append(nn3)

        nn4 = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('ME', self.dados)
        nn4.adicionar_camada(camada.Camada(len(self.dados.X_treino[0]), 2, 'tanh')) 
        nn4.adicionar_camada(camada.Camada(2, 10, 'sigmoid')) 
        nn4.adicionar_camada(camada.Camada(10, 4, 'tanh')) 
        nn4.adicionar_camada(camada.Camada(4, 1, 'sigmoid')) 
        lista_especialistas.append(nn4)

        nn5 = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('ME', self.dados)
        nn5.adicionar_camada(camada.Camada(len(self.dados.X_treino[0]), 2, 'tanh')) 
        nn5.adicionar_camada(camada.Camada(2, 10, 'sigmoid')) 
        nn5.adicionar_camada(camada.Camada(10, 4, 'tanh')) 
        nn5.adicionar_camada(camada.Camada(4, 1, 'sigmoid')) 
        lista_especialistas.append(nn5)

        nn6 = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('ME', self.dados)
        nn6.adicionar_camada(camada.Camada(len(self.dados.X_treino[0]), 2, 'tanh')) 
        nn6.adicionar_camada(camada.Camada(2, 10, 'sigmoid')) 
        nn6.adicionar_camada(camada.Camada(10, 4, 'tanh')) 
        nn6.adicionar_camada(camada.Camada(4, 1, 'sigmoid')) 
        lista_especialistas.append(nn6)

        nn7 = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('ME', self.dados)
        nn7.adicionar_camada(camada.Camada(len(self.dados.X_treino[0]), 2, 'tanh')) 
        nn7.adicionar_camada(camada.Camada(2, 10, 'sigmoid')) 
        nn7.adicionar_camada(camada.Camada(10, 4, 'tanh')) 
        nn7.adicionar_camada(camada.Camada(4, 1, 'sigmoid')) 
        lista_especialistas.append(nn7)

        nn8 = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('ME', self.dados)
        nn8.adicionar_camada(camada.Camada(len(self.dados.X_treino[0]), 2, 'tanh')) 
        nn8.adicionar_camada(camada.Camada(2, 10, 'sigmoid')) 
        nn8.adicionar_camada(camada.Camada(10, 4, 'tanh')) 
        nn8.adicionar_camada(camada.Camada(4, 1, 'sigmoid')) 
        lista_especialistas.append(nn8)


        # nn1 = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('ME', self.dados)
        # nn1.adicionar_camada(camada.Camada(len(self.dados.X_treino[0]), 2, 'tanh')) 
        # nn1.adicionar_camada(camada.Camada(2, 1, 'tanh')) 
        # lista_especialistas.append(nn1)

        # nn2 = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('ME', self.dados)
        # nn2.adicionar_camada(camada.Camada(len(self.dados.X_treino[0]), 5, 'sigmoid')) 
        # nn2.adicionar_camada(camada.Camada(5, 1, 'tanh')) 
        # lista_especialistas.append(nn2)

        # nn3 = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('ME', self.dados)
        # nn3.adicionar_camada(camada.Camada(len(self.dados.X_treino[0]), 2, 'tanh')) 
        # nn3.adicionar_camada(camada.Camada(2, 10, 'tanh')) 
        # nn3.adicionar_camada(camada.Camada(10, 4, 'tanh')) 
        # nn3.adicionar_camada(camada.Camada(4, 1, 'tanh')) 
        # lista_especialistas.append(nn3)
        
        m = mistura.Mistura_Especialista(lista_especialistas)
        return m.mistura(self.dados)

        # ME3 = me.MisturaEspecialista(lista_especialistas)

        # # Treinamento
        # erros_medios_ME3, erro_medio_pre_ME3, erro_medio_pos_ME3, saida_treinamento_ME3, tempo_ME3, predicao_ME3, mse_validacao_ME3, trace_erros_medios, trace_Treino, trace_validacao = self.treinar_rede_especialista(ME3)

        # return erros_medios_ME3, erro_medio_pre_ME3, erro_medio_pos_ME3, saida_treinamento_ME3, tempo_ME3, predicao_ME3, mse_validacao_ME3, trace_erros_medios, trace_Treino, trace_validacao

    def mistura_especialista_4(self):
        self.nome_rede = 'Mistura Especialista 4'
        # ME_B3 = '''
        # ##########################
        # ## MLP 4 camadas - Mistura Especialista
        # ## Camada 2:        2 Neurônios  - Tangente
        # ## Camada 10:       10 Neurônios - Sigmoide
        # ## Camada 4:        4 Neurônios  - Tangente
        # ## Camada Saída:    1 Neurônio   - Sigmoide
        # ##########################
        # '''


        # print(ME_B3 + "\n")

        lista_especialistas = []
        
        nn1 = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('ME', self.dados)
        nn1.adicionar_camada(camada.Camada(len(self.dados.X_treino[0]), 5, 'sigmoid')) # Linear
        nn1.adicionar_camada(camada.Camada(5, 1, 'tanh')) # Sigmoid
        lista_especialistas.append(nn1)

        nn2 = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('ME', self.dados)
        nn2.adicionar_camada(camada.Camada(len(self.dados.X_treino[0]), 3, 'tanh')) 
        nn2.adicionar_camada(camada.Camada(3, 10, 'sigmoid')) 
        nn2.adicionar_camada(camada.Camada(10, 1, 'sigmoid')) 
        lista_especialistas.append(nn2)

        nn3 = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('ME', self.dados)
        nn3.adicionar_camada(camada.Camada(len(self.dados.X_treino[0]), 2, 'tanh')) 
        nn3.adicionar_camada(camada.Camada(2, 10, 'tanh')) 
        nn3.adicionar_camada(camada.Camada(10, 4, 'tanh')) 
        nn3.adicionar_camada(camada.Camada(4, 1, 'tanh')) 
        lista_especialistas.append(nn3)

        nn_RE = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('recorrente', self.dados)
        nn_RE.adicionar_camada(camada.CamadaRecorrente(len(self.dados.X_treino[0]), 4, 'tanh', None, None, atraso=True))
        nn_RE.adicionar_camada(camada.CamadaRecorrente(4, 1, 'tanh', None, None, atraso=False))
        lista_especialistas.append(nn_RE)
        
        m = mistura.Mistura_Especialista(lista_especialistas)
        return m.mistura(self.dados)

        # ME3 = me.MisturaEspecialista(lista_especialistas)

        # # Treinamento
        # erros_medios_ME3, erro_medio_pre_ME3, erro_medio_pos_ME3, saida_treinamento_ME3, tempo_ME3, predicao_ME3, mse_validacao_ME3, trace_erros_medios, trace_Treino, trace_validacao = self.treinar_rede_especialista(ME3)

        # return erros_medios_ME3, erro_medio_pre_ME3, erro_medio_pos_ME3, saida_treinamento_ME3, tempo_ME3, predicao_ME3, mse_validacao_ME3, trace_erros_medios, trace_Treino, trace_validacao

############################################################################################################
############################################################################################################
############################################################################################################



    def rede_recorrente_1(self):
        self.nome_rede = 'Recorrente 1'
        RE_1 = '''
        ##########################
        ## Recorrente 1 camada
        ## Camada 1:        4 Neurônios  - Tangente
        ## Camada Saída:    1 Neurônio   - Tangente
        ##########################
        '''

        print(RE_1 + "\n")
        nn_RE = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('recorrente', self.dados)

        # Definir camadas da Rede
        nn_RE.adicionar_camada(camada.CamadaRecorrente(len(self.dados.X_treino[0]), 4, 'tanh', None, None, atraso=True))
        nn_RE.adicionar_camada(camada.CamadaRecorrente(4, 1, 'tanh', None, None, atraso=False))

        # Treina a Rede
        erros_medios_RE_1, erro_medio_pre_RE_1, erro_medio_pos_RE_1, saida_treinamento_RE_1, tempo_RE_1, predicao_RE_1, mse_validacao_RE_1, trace_erros_medios_RE_1, trace_Treino_RE_1, trace_validacao_RE_1 = self.treinar_rede(nn_RE)

        return erros_medios_RE_1, erro_medio_pre_RE_1, erro_medio_pos_RE_1, saida_treinamento_RE_1, tempo_RE_1, predicao_RE_1, mse_validacao_RE_1, trace_erros_medios_RE_1, trace_Treino_RE_1, trace_validacao_RE_1

        ##############################################################################

    def rede_recorrente_2(self):
        self.nome_rede = 'Recorrente 2'
        RE_1 = '''
        ##########################
        ## Recorrente 3 camada
        ## Camada 1:        4 Neurônios  - Tangente
        ## Camada 2:        4 Neurônios  - Tangente
        ## Camada Saída:    1 Neurônio   - Tangente
        ##########################
        '''

        print(RE_1 + "\n")
        nn_RE = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('recorrente', self.dados)

        # Definir camadas da Rede
        nn_RE.adicionar_camada(camada.CamadaRecorrente(len(self.dados.X_treino[0]), 4, 'tanh', None, None, atraso=True))
        nn_RE.adicionar_camada(camada.CamadaRecorrente(4, 4, 'tanh', None, None, atraso=True))
        nn_RE.adicionar_camada(camada.CamadaRecorrente(4, 1, 'tanh', None, None, atraso=False))

        # Treina a Rede
        erros_medios_RE_1, erro_medio_pre_RE_1, erro_medio_pos_RE_1, saida_treinamento_RE_1, tempo_RE_1, predicao_RE_1, mse_validacao_RE_1, trace_erros_medios_RE_1, trace_Treino_RE_1, trace_validacao_RE_1 = self.treinar_rede(nn_RE)

        return erros_medios_RE_1, erro_medio_pre_RE_1, erro_medio_pos_RE_1, saida_treinamento_RE_1, tempo_RE_1, predicao_RE_1, mse_validacao_RE_1, trace_erros_medios_RE_1, trace_Treino_RE_1, trace_validacao_RE_1

        ##############################################################################        

    def rede_recorrente_3(self):
        self.nome_rede = 'Recorrente 3'
        RE_1 = '''
        ##########################
        ## Recorrente 2 camada
        ## Camada 1:        10 Neurônios  - Tangente
        ## Camada Saída:    1 Neurônio   - Tangente
        ##########################
        '''

        print(RE_1 + "\n")
        nn_RE = rn_mlp_ME.RedeNeural_MLP().criar_rede_neural('recorrente', self.dados)

        # Definir camadas da Rede
        nn_RE.adicionar_camada(camada.CamadaRecorrente(len(self.dados.X_treino[0]), 10, 'tanh', None, None, atraso=True))
        nn_RE.adicionar_camada(camada.CamadaRecorrente(10, 1, 'tanh', None, None, atraso=False))

        # Treina a Rede
        erros_medios_RE_1, erro_medio_pre_RE_1, erro_medio_pos_RE_1, saida_treinamento_RE_1, tempo_RE_1, predicao_RE_1, mse_validacao_RE_1, trace_erros_medios_RE_1, trace_Treino_RE_1, trace_validacao_RE_1 = self.treinar_rede(nn_RE)

        return erros_medios_RE_1, erro_medio_pre_RE_1, erro_medio_pos_RE_1, saida_treinamento_RE_1, tempo_RE_1, predicao_RE_1, mse_validacao_RE_1, trace_erros_medios_RE_1, trace_Treino_RE_1, trace_validacao_RE_1

        ############################################################################## 
