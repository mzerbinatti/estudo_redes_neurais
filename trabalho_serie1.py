import numpy as np
import matplotlib.pyplot as plt
import redeneuralME as rn_mlp_ME
import camadaRN as camada
# import mistura as me
import auxiliar as manipulacao
import redesneuraisconfiguradas as rnc
import dados as dados_serie
import time
# import mistura_prof_michel as mistura

import plotly
import plotly.graph_objs as go



def gerar_graficos_normalizacoes():
    for i in range(4):
        grafico_normalizacao(i + 1)

def carregar_serie(num_serie):
    nome_arquivo_serie = "dados/serie" + str(num_serie) + "_trein.txt"
    serie = np.array(manipulacao.ManipulacaoDados.carregar_arquivo(nome_arquivo_serie))
    return serie

def grafico_normalizacao(num_serie):
    serie = carregar_serie(num_serie)
    serie = serie[0:65] # Pega somente os primeiros 65 registros
    serie_normalizada = manipulacao.ManipulacaoDados.normalizar_dados(serie)
    serie_standardize = manipulacao.ManipulacaoDados.standardize_dados(serie)
    # serie_standardize = manipulacao.ManipulacaoDados.normalizacao_desvio(serie)

    nome_arquivo_html = "tipos_normalizacao_" + str(num_serie) + ".html"
    titulo = "Série e Normalizações - Série "  +  str(num_serie) + " (Primeiros 65 registros)"
    tamanho = len(serie)

    trace_orig = go.Scatter(
        x = np.array(list(range(tamanho))),
        y = np.array(serie.reshape(-1)),
        name= "Original",
        mode= "lines+markers"
    )
    trace_norm = go.Scatter(
        x = np.array(list(range(tamanho))),
        y = np.array(serie_normalizada.reshape(-1)),
        name= "Normalização (0 e 1)",
        mode= "lines+markers"
    )
    trace_stan = go.Scatter(
        x = np.array(list(range(tamanho))),
        y = np.array(serie_standardize.reshape(-1)),
        name= "Standardize - Desvio Padrão",
        mode= "lines+markers"
    )

    dados_normalizacao  = [trace_orig, trace_norm, trace_stan]
    
    plotly.offline.plot({
        "data": dados_normalizacao,
        "layout": go.Layout(
            title=titulo,
            xaxis=dict(
                title="t",
                showticklabels=True,
                dtick=1
            ),
            yaxis=dict(
                title="X"
            )
        )
    }, auto_open=True, filename= nome_arquivo_html)


def gerar_graficos_autocorrelacao_lags():
    for i in range(4):
        grafico_autocorrelacao(i + 1)

def grafico_autocorrelacao(num_serie):
    serie = carregar_serie(num_serie)
    correlacao_lags = manipulacao.ManipulacaoDados.calcular_lag(serie)

    nome_arquivo_html = "autocorrelacao_serie" + str(num_serie) + ".html"
    titulo = "Autocorrelação - Serie "  +  str(num_serie)

    trace_autocorrelacao = go.Scatter(
        x = np.array(list(range(len(correlacao_lags)))),
        y = np.array(correlacao_lags.reshape(-1)),
        name= "Autocorrelação",
        mode= "lines+markers"
    )
    dados_autocorrelacao = [trace_autocorrelacao]
    plotly.offline.plot({
        "data": dados_autocorrelacao,
        "layout": go.Layout(
            title=titulo,
            xaxis=dict(
                title="Lag",
                showticklabels=True,
                dtick=1
            ),
            yaxis=dict(
                title="Autocorrelação"
            )
        )
    }, auto_open=True, filename= nome_arquivo_html)

def gerar_graficos_analises(num_serie, tipo_rede, dados_erros_medios, dados_treinamento, dados_validacao):
    from datetime import datetime
    now = datetime.now() # current date and time
    data_hora = now.strftime("%Y%m%d_%H_%M")

    # EQM
    plotly.offline.plot({
        "data": dados_erros_medios,
        "layout": go.Layout(
            title= tipo_rede + " - EQM",
            xaxis=dict(
                title="Épocas"
            ),
            yaxis=dict(
                title="EQM"
            )
        )
    }, auto_open=True, filename= "erro_medio_" + tipo_rede.replace(" ", "_") + "_" + data_hora + "_" + str(num_serie) + ".html")

    # Treinamento
    plotly.offline.plot({
        "data": dados_treinamento,
        "layout": go.Layout(
            title="Treinamento",
            xaxis=dict(
                title="Épocas"
            ),
            yaxis=dict(
                title="Y"
            )
        )
    }, auto_open=True, filename= "treinamento_" + tipo_rede.replace(" ", "_") + "_" + data_hora + "_" + str(num_serie) + ".html")

    # Validação
    plotly.offline.plot({
        "data": dados_validacao,
        "layout": go.Layout(
            title="Validacao",
            xaxis=dict(
                title="Épocas"
            ),
            yaxis=dict(
                title="Y"
            )
        )
    }, auto_open=True, filename= "validacao_" + tipo_rede.replace(" ", "_") + "_" + data_hora + "_" + str(num_serie) + ".html")

def gerar_grafico_divisao_especialistas(num_serie, tipo_serie,  especialistas):
    from datetime import datetime
    now = datetime.now() # current date and time
    data_hora = now.strftime("%Y%m%d_%H_%M")

    total = np.sum(especialistas, 0)[1]

    linhas = especialistas.shape[0]
    nome_espec = [] # np.zeros((linhas, 1))
    percentual_espec = [] # np.array((linhas, 1))
    for i in range(especialistas.shape[0]):
        nome_espec.append("Especialista " + str(int(especialistas[i,0])))
        percentual_espec.append(round((100 * especialistas[i,1] / total), 1))

    data = [
        go.Bar(
                x=nome_espec,
                y=percentual_espec,
                text=percentual_espec,
                textposition = 'auto'
        )]

    layout = go.Layout(
        xaxis=dict(tickangle=-0),
        barmode='group',
        title="Percentual de cada Especialista - " + tipo_serie
    )

    plotly.offline.plot({
        "data": data,
        "layout": layout
    }, auto_open=True, filename= "Percentual_Especialistas_" + tipo_serie + "_" + data_hora + "_" + str(num_serie) + ".html")


def gerar_grafico_eqm_fases(num_serie, tipo_rede, erros_pre_treinamento, erros_pos_treinamento, erros_validacao):
    from datetime import datetime
    now = datetime.now() # current date and time
    data_hora = now.strftime("%Y%m%d_%H_%M")

    
    y_pre = [round(x,5) for x in erros_pre_treinamento] 
    trace_pre = go.Bar(
        x=[tipo_rede + ' 1', tipo_rede + ' 2', tipo_rede + ' 3'],
        y=y_pre,
        text=y_pre,
        textposition = 'auto',
        name='EQM Pré Treinamento',
        marker=dict(
            color='rgb(255,0,0)',
            line=dict(
                color='rgb(255,0,0)',
                width=1.5
            ),
            opacity=0.8
        )
    )

   
    y_pos = [round(x,5) for x in erros_pos_treinamento] 
    trace_pos = go.Bar(
        x=[tipo_rede + ' 1', tipo_rede + ' 2', tipo_rede + ' 3'],
        y=y_pos,
        text=y_pos,
        textposition = 'auto',
        name='EQM Pós Treinamento',
        marker=dict(
            color='rgb(0,128,0)',
            line=dict(
                color='rgb(0,128,0)',
                width=1.5
            ),
            opacity=0.8
        )
    )

    
    y_val = [round(x,5) for x in erros_validacao] 
    trace_val = go.Bar(
        x=[tipo_rede + ' 1', tipo_rede + ' 2', tipo_rede + ' 3'],
        y=y_val,
        text=y_val,
        textposition = 'auto',
        name='EQM Validação',
        marker=dict(
            color='rgb(0,102,204)',
            line=dict(
                color='rgb(0,102,204)',
                width=1.5
            ),
            opacity=0.8
        )
    )

    dados_eqm = [trace_pre, trace_pos, trace_val]
    layout = go.Layout(
        xaxis=dict(tickangle=-0),
        barmode='group',
        title="EQM - " + tipo_rede
    )
    plotly.offline.plot({
        "data": dados_eqm,
        "layout": layout
    }, auto_open=True, filename= "EQM_comparacao_" + tipo_rede + "_" + data_hora + "_" + str(num_serie) + ".html")


def gerar_grafico_tempo(num_serie, tipo_rede, tempo_execucao):
    from datetime import datetime
    now = datetime.now() # current date and time
    data_hora = now.strftime("%Y%m%d_%H_%M")
    
    y_tempo = [round(x,2) for x in tempo_execucao] 
    trace_tempo = go.Bar(
        x=[tipo_rede + ' 1', tipo_rede + ' 2', tipo_rede + ' 3'],
        y=y_tempo,
        text=y_tempo,
        textposition = 'auto',
        name='EQM Pré Treinamento',
        marker=dict(
            color='rgb(0,102,204)',
            line=dict(
                color='rgb(0,102,204)',
                width=1.5
            ),
            opacity=0.8
        )
    )

    dados_tempo = [trace_tempo]
    layout = go.Layout(
        xaxis=dict(tickangle=-0),
        yaxis=dict(range=[0, 600]),
        barmode='group',
        title="Tempo Execução (segundos) - " + tipo_rede
    )

    plotly.offline.plot({
        "data": dados_tempo,
        "layout": layout
    }, auto_open=True, filename= "tempo_execucao_" + tipo_rede + "_" + data_hora + "_" + str(num_serie) + ".html")



def executar_series(num_serie):
    tempo_total_inicial = time.time()

    #########################################################################################################
    # Exibe os gráficos de normalização das séries
    # gerar_graficos_normalizacoes()

    # Exibe gráficos dos LAGs/Autocorrelação das Séries
    # gerar_graficos_autocorrelacao_lags()
    #########################################################################################################

    # np.random.seed(10)
    max_epocas = 10000
    taxa_aprendizagem = 0.01

    # serie1 = 12 - 0.90
    # serie2 = 12 - 0.97
    # serie3 = 12 - 0.90
    # serie4 = 1  - 0.69 | 12 - 0.63
    # serie_txt = np.array(manipulacao.ManipulacaoDados.carregar_arquivo('dados/serie3_trein.txt'))
    serie_txt = carregar_serie(num_serie)
    # serie_normalizada = manipulacao.ManipulacaoDados.normalizar_dados(serie_txt)
    # lag_utilizado = 12
    # X, Y = manipulacao.ManipulacaoDados.preparar_array_X_Y(serie_normalizada, lag_utilizado)
    # Tamanho_X = len(X)
    # Tamanho_Treino = int(Tamanho_X * 0.7)
    # Y = np.reshape(Y, (len(Y), 1))
    # X_treino = X[0:Tamanho_Treino]
    # Y_treino = Y[0:Tamanho_Treino]
    # X_validacao = X[Tamanho_Treino + 1:]
    # Y_validacao = Y[Tamanho_Treino + 1:]

    dados = dados_serie.Dados(serie_txt, 12, 'minmax', 0.7, 0, 1)

    time_inicio_total = time.time()

    # rn = rnc.RedesNeuraisConfiguradas(X_treino, Y_treino, X_validacao, Y_validacao, taxa_aprendizagem, max_epocas) 
    rn = rnc.RedesNeuraisConfiguradas(dados, taxa_aprendizagem, max_epocas) 

    erros_medios_MLP_B1, erro_medio_pre_MLP_B1, erro_medio_pos_MLP_B1, saida_treinamento_MLP_B1, tempo_MLP_B1, predicao_MLP_B1, mse_validacao_MLP_B1, trace_erros_medios_MLP_B1, trace_Treino_MLP_B1, trace_validacao_MLP_B1 = rn.mlp_batch_1()
    erros_medios_MLP_B2, erro_medio_pre_MLP_B2, erro_medio_pos_MLP_B2, saida_treinamento_MLP_B2, tempo_MLP_B2, predicao_MLP_B2, mse_validacao_MLP_B2, trace_erros_medios_MLP_B2, trace_Treino_MLP_B2, trace_validacao_MLP_B2 = rn.mlp_batch_2()
    erros_medios_MLP_B3, erro_medio_pre_MLP_B3, erro_medio_pos_MLP_B3, saida_treinamento_MLP_B3, tempo_MLP_B3, predicao_MLP_B3, mse_validacao_MLP_B3, trace_erros_medios_MLP_B3, trace_Treino_MLP_B3, trace_validacao_MLP_B3 = rn.mlp_batch_3()

    erros_medios_MLP_E1, erro_medio_pre_MLP_E1, erro_medio_pos_MLP_E1, saida_treinamento_MLP_E1, tempo_MLP_E1, predicao_MLP_E1, mse_validacao_MLP_E1, trace_erros_medios_MLP_E1, trace_Treino_MLP_E1, trace_validacao_MLP_E1 = rn.mlp_estocastico_1()
    erros_medios_MLP_E2, erro_medio_pre_MLP_E2, erro_medio_pos_MLP_E2, saida_treinamento_MLP_E2, tempo_MLP_E2, predicao_MLP_E2, mse_validacao_MLP_E2, trace_erros_medios_MLP_E2, trace_Treino_MLP_E2, trace_validacao_MLP_E2 = rn.mlp_estocastico_2()
    erros_medios_MLP_E3, erro_medio_pre_MLP_E3, erro_medio_pos_MLP_E3, saida_treinamento_MLP_E3, tempo_MLP_E3, predicao_MLP_E3, mse_validacao_MLP_E3, trace_erros_medios_MLP_E3, trace_Treino_MLP_E3, trace_validacao_MLP_E3 = rn.mlp_estocastico_3()

    erros_medios_ME1, erro_medio_pre_ME1, erro_medio_pos_ME1, saida_treinamento_ME1, tempo_ME1, predicao_ME1, mse_validacao_ME1, especilista_treino_ME1, especilista_valid_ME1, especialista_lider_treino_ME1, especialista_lider_valid_ME1, trace_erros_medios_ME1, trace_Treino_ME1, trace_validacao_ME1  = rn.mistura_especialista_1()
    erros_medios_ME2, erro_medio_pre_ME2, erro_medio_pos_ME2, saida_treinamento_ME2, tempo_ME2, predicao_ME2, mse_validacao_ME2, especilista_treino_ME2, especilista_valid_ME2, especialista_lider_treino_ME2, especialista_lider_valid_ME2, trace_erros_medios_ME2, trace_Treino_ME2, trace_validacao_ME2  = rn.mistura_especialista_2()
    erros_medios_ME3, erro_medio_pre_ME3, erro_medio_pos_ME3, saida_treinamento_ME3, tempo_ME3, predicao_ME3, mse_validacao_ME3, especilista_treino_ME3, especilista_valid_ME3, especialista_lider_treino_ME3, especialista_lider_valid_ME3, trace_erros_medios_ME3, trace_Treino_ME3, trace_validacao_ME3  = rn.mistura_especialista_3()
    erros_medios_ME4, erro_medio_pre_ME4, erro_medio_pos_ME4, saida_treinamento_ME4, tempo_ME4, predicao_ME4, mse_validacao_ME4, especilista_treino_ME4, especilista_valid_ME4, especialista_lider_treino_ME4, especialista_lider_valid_ME4, trace_erros_medios_ME4, trace_Treino_ME4, trace_validacao_ME4  = rn.mistura_especialista_4()

    erros_medios_RE1, erro_medio_pre_RE1, erro_medio_pos_RE1, saida_treinamento_RE1, tempo_RE1, predicao_RE1, mse_validacao_RE1, trace_erros_medios_RE1, trace_Treino_RE1, trace_validacao_RE1 = rn.rede_recorrente_1()
    erros_medios_RE2, erro_medio_pre_RE2, erro_medio_pos_RE2, saida_treinamento_RE2, tempo_RE2, predicao_RE2, mse_validacao_RE2, trace_erros_medios_RE2, trace_Treino_RE2, trace_validacao_RE2 = rn.rede_recorrente_2()
    erros_medios_RE3, erro_medio_pre_RE3, erro_medio_pos_RE3, saida_treinamento_RE3, tempo_RE3, predicao_RE3, mse_validacao_RE3, trace_erros_medios_RE3, trace_Treino_RE3, trace_validacao_RE3 = rn.rede_recorrente_3()

    time_fim_total = time.time()
    time_total = time_fim_total - time_inicio_total

    print("Tempo Execução TOTAL: ", time_total)
    ############### 

    N_Treino = len(dados.X_treino)
    qtd_treino  = np.array(list(range(N_Treino))) + 1

    dados.X_validacao = dados.desnormalizar(dados.X_validacao)
    N_Validacao = len(dados.X_validacao)
    qtd_validacao = np.array(list(range(N_Validacao))) + 1

    qtd_epocas = np.array(list(range(max_epocas))) + 1

    Y_original_treino = dados.desnormalizar(dados.Y_treino)
    Y_original_treino = Y_original_treino.reshape(-1)

    Y_original_validacao = dados.desnormalizar(dados.Y_validacao)
    Y_original_validacao = Y_original_validacao.reshape(-1)


    # ERROS MEDIOS
    erros_zerados = np.zeros((max_epocas,1))
    trace_erro_zerado = go.Scatter(
        x = qtd_treino,
        y = erros_zerados.reshape(-1),
        name= ' - '
    )

    # TREINAMENTO
    trace_Treino = go.Scatter(
        x = qtd_treino,
        y = Y_original_treino,
        name= 'Y Esperado'
    )

    # Validação
    trace_Validacao = go.Scatter(
        x = qtd_validacao,
        y = Y_original_validacao,
        name= 'Y Esperado'
    )

    # # dados_erros_medios = [trace_erro_zerado, trace_erros_medios_MLP_B1, trace_erros_medios_MLP_B2, trace_erros_medios_MLP_B3, trace_erros_medios_MLP_E1, trace_erros_medios_MLP_E2, trace_erros_medios_MLP_E3, trace_erros_medios_ME1, trace_erros_medios_ME2, trace_erros_medios_ME3, trace_erros_medios_RE1, trace_erros_medios_RE2, trace_erros_medios_RE3] 
    # dados_erros_medios = [trace_erro_zerado, trace_erros_medios_MLP_B1, trace_erros_medios_MLP_B2, trace_erros_medios_MLP_B3] 

    # # dados_treinamento = [trace_Treino, trace_Treino_MLP_B1, trace_Treino_MLP_B2, trace_Treino_MLP_B3, trace_Treino_MLP_E1, trace_Treino_MLP_E2, trace_Treino_MLP_E3, trace_Treino_ME1, trace_Treino_ME2, trace_Treino_ME3, trace_Treino_RE1, trace_Treino_RE2, trace_Treino_RE3] 
    # dados_treinamento = [trace_Treino, trace_Treino_MLP_B1, trace_Treino_MLP_B2, trace_Treino_MLP_B3] 

    # # dados_validacao = [trace_Validacao, trace_validacao_MLP_B1, trace_validacao_MLP_B2, trace_validacao_MLP_B3, trace_validacao_MLP_E1, trace_validacao_MLP_E2, trace_validacao_MLP_E3, trace_validacao_ME1, trace_validacao_ME2, trace_validacao_ME3, trace_validacao_RE1, trace_validacao_RE2, trace_validacao_RE3] 
    # dados_validacao = [trace_Validacao, trace_validacao_MLP_B1, trace_validacao_MLP_B2, trace_validacao_MLP_B3] 

    #########################################################################################


    ######################################################################################################################
    # BATCH
    ######################################################################################################################
    # EQM através das Épocas
    dados_erros_medios_batch = [trace_erro_zerado, trace_erros_medios_MLP_B1, trace_erros_medios_MLP_B2, trace_erros_medios_MLP_B3] 
    dados_treinamento_batch = [trace_Treino, trace_Treino_MLP_B1, trace_Treino_MLP_B2, trace_Treino_MLP_B3] 
    dados_validacao_batch = [trace_Validacao, trace_validacao_MLP_B1, trace_validacao_MLP_B2, trace_validacao_MLP_B3] 
    gerar_graficos_analises(num_serie, "Batch", dados_erros_medios_batch, dados_treinamento_batch, dados_validacao_batch)

    # EQM Pre X EQM Pos X EQM Val
    eqm_pre_batch = [erro_medio_pre_MLP_B1, erro_medio_pre_MLP_B2, erro_medio_pre_MLP_B3]
    eqm_pos_batch = [erro_medio_pos_MLP_B1, erro_medio_pos_MLP_B2, erro_medio_pos_MLP_B3]
    eqm_val_batch = [mse_validacao_MLP_B1, mse_validacao_MLP_B2, mse_validacao_MLP_B3]
    gerar_grafico_eqm_fases(num_serie, "MLP Batch", eqm_pre_batch, eqm_pos_batch, eqm_val_batch)

    # Tempo de Execução
    tempo_batch = [tempo_MLP_B1, tempo_MLP_B2, tempo_MLP_B3]
    gerar_grafico_tempo(num_serie, "Batch", tempo_batch)


    ######################################################################################################################
    # ESTOCASTICO
    ######################################################################################################################
    # EQM através das Épocas
    dados_erros_medios_estocastico = [trace_erro_zerado, trace_erros_medios_MLP_E1, trace_erros_medios_MLP_E2, trace_erros_medios_MLP_E3] 
    dados_treinamento_estocastico = [trace_Treino, trace_Treino_MLP_E1, trace_Treino_MLP_E2, trace_Treino_MLP_E3] 
    dados_validacao_estocastico = [trace_Validacao, trace_validacao_MLP_E1, trace_validacao_MLP_E2, trace_validacao_MLP_E3] 
    gerar_graficos_analises(num_serie, "Estocastico", dados_erros_medios_estocastico, dados_treinamento_estocastico, dados_validacao_estocastico)

    # EQM Pre X EQM Pos X EQM Val
    eqm_pre_estoc = [erro_medio_pre_MLP_E1, erro_medio_pre_MLP_E2, erro_medio_pre_MLP_E3]
    eqm_pos_estoc = [erro_medio_pos_MLP_E1, erro_medio_pos_MLP_E2, erro_medio_pos_MLP_E3]
    eqm_val_estoc = [mse_validacao_MLP_E1, mse_validacao_MLP_E2, mse_validacao_MLP_E3]
    gerar_grafico_eqm_fases(num_serie, "MLP Estocástico", eqm_pre_estoc, eqm_pos_estoc, eqm_val_estoc)

    # Tempo de Execução
    tempo_estoc = [tempo_MLP_E1, tempo_MLP_E2, tempo_MLP_E3]
    gerar_grafico_tempo(num_serie, "Estocástico", tempo_estoc)

    ######################################################################################################################
    # MISTURA DE ESPEECIALISTAS A
    ######################################################################################################################
    # EQM através das Épocas
    dados_erros_medios_me1 = [trace_erros_medios_ME1] 
    dados_treinamento_me1 = [trace_Treino, trace_Treino_ME1] 
    for i in range(len(especilista_treino_ME1)):
        # Dados Plot Erro Médio Quadratico
        trace_Treino_espec_me1 = go.Scatter(
            x = np.array(list(range(especilista_treino_ME1[i].shape[0]))) + 1,
            y = np.array(especilista_treino_ME1[i]).reshape(-1),
            name = "Especialista " + str(i + 1)
        )
        dados_treinamento_me1.append(trace_Treino_espec_me1)
    dados_validacao_me1 = [trace_Validacao, trace_validacao_ME1] 
    for i in range(len(especilista_valid_ME1)):
        # Dados Plot Erro Médio Quadratico
        trace_Validacao_espec_me1 = go.Scatter(
            x = np.array(list(range(especilista_valid_ME1[i].shape[0]))) + 1,
            y = np.array(especilista_valid_ME1[i]).reshape(-1),
            name = "Especialista " + str(i + 1)
        )
        dados_validacao_me1.append(trace_Validacao_espec_me1)
    gerar_graficos_analises(num_serie, "Mistura de Especialistas A", dados_erros_medios_me1, dados_treinamento_me1, dados_validacao_me1)


    # EQM Pre X EQM Pos X EQM Val
    eqm_pre_me1 = [erro_medio_pre_ME1]
    eqm_pos_me1 = [erro_medio_pos_ME1]
    eqm_val_me1 = [mse_validacao_ME1]
    gerar_grafico_eqm_fases(num_serie, "Mistura Especialista A", eqm_pre_me1, eqm_pos_me1, eqm_val_me1)

    # Tempo de Execução
    tempo_me1 = [tempo_ME1]
    gerar_grafico_tempo(num_serie, "Mistura de Especialista A", tempo_me1)

    # # Divisão dos Especilistas
    gerar_grafico_divisao_especialistas(num_serie, "Treinamento A", especialista_lider_treino_ME1)
    gerar_grafico_divisao_especialistas(num_serie, "Validação A", especialista_lider_valid_ME1)

    ######################################################################################################################
    # MISTURA DE ESPEECIALISTAS B
    ######################################################################################################################
    # EQM através das Épocas
    dados_erros_medios_ME2 = [trace_erros_medios_ME2] 
    dados_treinamento_ME2 = [trace_Treino, trace_Treino_ME2] 
    for i in range(len(especilista_treino_ME2)):
        # Dados Plot Erro Médio Quadratico
        trace_Treino_espec_ME2 = go.Scatter(
            x = np.array(list(range(especilista_treino_ME2[i].shape[0]))) + 1,
            y = np.array(especilista_treino_ME2[i]).reshape(-1),
            name = "Especialista " + str(i + 1)
        )
        dados_treinamento_ME2.append(trace_Treino_espec_ME2)
    dados_validacao_ME2 = [trace_Validacao, trace_validacao_ME2] 
    for i in range(len(especilista_valid_ME2)):
        # Dados Plot Erro Médio Quadratico
        trace_Validacao_espec_ME2 = go.Scatter(
            x = np.array(list(range(especilista_valid_ME2[i].shape[0]))) + 1,
            y = np.array(especilista_valid_ME2[i]).reshape(-1),
            name = "Especialista " + str(i + 1)
        )
        dados_validacao_ME2.append(trace_Validacao_espec_ME2)
    gerar_graficos_analises(num_serie, "Mistura de Especialistas B", dados_erros_medios_ME2, dados_treinamento_ME2, dados_validacao_ME2)


    # EQM Pre X EQM Pos X EQM Val
    eqm_pre_ME2 = [erro_medio_pre_ME2]
    eqm_pos_ME2 = [erro_medio_pos_ME2]
    eqm_val_ME2 = [mse_validacao_ME2]
    gerar_grafico_eqm_fases(num_serie, "Mistura Especialista B", eqm_pre_ME2, eqm_pos_ME2, eqm_val_ME2)

    # Tempo de Execução
    tempo_ME2 = [tempo_ME2]
    gerar_grafico_tempo(num_serie, "Mistura de Especialista B", tempo_ME2)

    # # Divisão dos Especilistas
    gerar_grafico_divisao_especialistas(num_serie, "Treinamento B", especialista_lider_treino_ME2)
    gerar_grafico_divisao_especialistas(num_serie, "Validação B", especialista_lider_valid_ME2)

    ######################################################################################################################
    # MISTURA DE ESPEECIALISTAS C
    ######################################################################################################################
    # EQM através das Épocas
    dados_erros_medios_ME3 = [trace_erros_medios_ME3] 
    dados_treinamento_ME3 = [trace_Treino, trace_Treino_ME3] 
    for i in range(len(especilista_treino_ME3)):
        # Dados Plot Erro Médio Quadratico
        trace_Treino_espec_ME3 = go.Scatter(
            x = np.array(list(range(especilista_treino_ME3[i].shape[0]))) + 1,
            y = np.array(especilista_treino_ME3[i]).reshape(-1),
            name = "Especialista " + str(i + 1)
        )
        dados_treinamento_ME3.append(trace_Treino_espec_ME3)
    dados_validacao_ME3 = [trace_Validacao, trace_validacao_ME3] 
    for i in range(len(especilista_valid_ME3)):
        # Dados Plot Erro Médio Quadratico
        trace_Validacao_espec_ME3 = go.Scatter(
            x = np.array(list(range(especilista_valid_ME3[i].shape[0]))) + 1,
            y = np.array(especilista_valid_ME3[i]).reshape(-1),
            name = "Especialista " + str(i + 1)
        )
        dados_validacao_ME3.append(trace_Validacao_espec_ME3)
    gerar_graficos_analises(num_serie, "Mistura de Especialistas C", dados_erros_medios_ME3, dados_treinamento_ME3, dados_validacao_ME3)


    # EQM Pre X EQM Pos X EQM Val
    eqm_pre_ME3 = [erro_medio_pre_ME3]
    eqm_pos_ME3 = [erro_medio_pos_ME3]
    eqm_val_ME3 = [mse_validacao_ME3]
    gerar_grafico_eqm_fases(num_serie, "Mistura Especialista C", eqm_pre_ME3, eqm_pos_ME3, eqm_val_ME3)

    # Tempo de Execução
    tempo_ME3 = [tempo_ME3]
    gerar_grafico_tempo(num_serie, "Mistura de Especialista C", tempo_ME3)

    # # Divisão dos Especilistas
    gerar_grafico_divisao_especialistas(num_serie, "Treinamento C", especialista_lider_treino_ME3)
    gerar_grafico_divisao_especialistas(num_serie, "Validação C", especialista_lider_valid_ME3)


    ######################################################################################################################
    # MISTURA DE ESPEECIALISTAS D
    ######################################################################################################################
    # EQM através das Épocas
    dados_erros_medios_ME4 = [trace_erros_medios_ME4] 
    dados_treinamento_ME4 = [trace_Treino, trace_Treino_ME4] 
    for i in range(len(especilista_treino_ME4)):
        # Dados Plot Erro Médio Quadratico
        trace_Treino_espec_ME4 = go.Scatter(
            x = np.array(list(range(especilista_treino_ME4[i].shape[0]))) + 1,
            y = np.array(especilista_treino_ME4[i]).reshape(-1),
            name = "Especialista " + str(i + 1)
        )
        dados_treinamento_ME4.append(trace_Treino_espec_ME4)
    dados_validacao_ME4 = [trace_Validacao, trace_validacao_ME4] 
    for i in range(len(especilista_valid_ME4)):
        # Dados Plot Erro Médio Quadratico
        trace_Validacao_espec_ME4 = go.Scatter(
            x = np.array(list(range(especilista_valid_ME4[i].shape[0]))) + 1,
            y = np.array(especilista_valid_ME4[i]).reshape(-1),
            name = "Especialista " + str(i + 1)
        )
        dados_validacao_ME4.append(trace_Validacao_espec_ME4)
    gerar_graficos_analises(num_serie, "Mistura de Especialistas D", dados_erros_medios_ME4, dados_treinamento_ME4, dados_validacao_ME4)


    # EQM Pre X EQM Pos X EQM Val
    eqm_pre_ME4 = [erro_medio_pre_ME4]
    eqm_pos_ME4 = [erro_medio_pos_ME4]
    eqm_val_ME4 = [mse_validacao_ME4]
    gerar_grafico_eqm_fases(num_serie, "Mistura Especialista D", eqm_pre_ME4, eqm_pos_ME4, eqm_val_ME4)

    # Tempo de Execução
    tempo_ME4 = [tempo_ME4]
    gerar_grafico_tempo(num_serie, "Mistura de Especialista D", tempo_ME4)

    # # Divisão dos Especilistas
    gerar_grafico_divisao_especialistas(num_serie, "Treinamento D", especialista_lider_treino_ME4)
    gerar_grafico_divisao_especialistas(num_serie, "Validação D", especialista_lider_valid_ME4)

    # ######################################################################################################################
    # # RECORRENTE
    # ######################################################################################################################
    # # EQM através das Épocas
    # dados_erros_medios_recorrente = [trace_erro_zerado, trace_erros_medios_RE1, trace_erros_medios_RE2, trace_erros_medios_RE3] 
    # dados_treinamento_recorrente = [trace_Treino, trace_Treino_RE1, trace_Treino_RE2, trace_Treino_RE3] 
    # dados_validacao_recorrente = [trace_Validacao, trace_validacao_RE1, trace_validacao_RE2, trace_validacao_RE3] 
    # gerar_graficos_analises(num_serie, "Recorrente", dados_erros_medios_recorrente, dados_treinamento_recorrente, dados_validacao_recorrente)

    # # EQM Pre X EQM Pos X EQM Val
    # eqm_pre_recor = [erro_medio_pre_RE1, erro_medio_pre_RE2, erro_medio_pre_RE3]
    # eqm_pos_recor = [erro_medio_pos_RE1, erro_medio_pos_RE2, erro_medio_pos_RE3]
    # eqm_val_recor = [mse_validacao_RE1, mse_validacao_RE2, mse_validacao_RE3]
    # gerar_grafico_eqm_fases(num_serie, "Recorrente", eqm_pre_recor, eqm_pos_recor, eqm_val_recor)

    # # Tempo de Execução
    # tempo_recor = [tempo_RE1, tempo_RE2, tempo_RE3]
    # gerar_grafico_tempo(num_serie, "Recorrente", tempo_recor)

    # tempo_total_final = time.time()

    # print("Tempo total de execução: " + str(tempo_total_final - tempo_total_inicial))


executar_series(1)
executar_series(2)
executar_series(3)
executar_series(4)

