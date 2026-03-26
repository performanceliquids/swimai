import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import fitparse
from datetime import timedelta
import google.generativeai as genai
from PIL import Image # Nova biblioteca para processar imagem

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Performance Liquids - SwimAI", layout="wide", page_icon="🌊", initial_sidebar_state="expanded")

# --- CSS MINIMALISTA (Esconde marcas d'água, mantém o Dark Mode) ---
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- MEMÓRIA ---
if 'biblioteca_series' not in st.session_state: st.session_state.biblioteca_series = {}
if 'treinos_usados' not in st.session_state: st.session_state.treinos_usados = set()
if 'perfil_atleta' not in st.session_state: st.session_state.perfil_atleta = {"nome": "", "nivel": "Avançado", "objetivo": "Polimento para Guaratuba"}
if 'provas_cadastradas' not in st.session_state: st.session_state.provas_cadastradas = []

# --- FUNÇÕES DE UTILIDADE ---
def format_pace(segundos):
    if pd.isna(segundos) or segundos <= 0: return "N/A"
    return f"{int(segundos//60)}:{int(segundos%60):02d}"

def formatar_eixo_y_pace_absoluto(fig, df_coluna):
    min_sec, max_sec = df_coluna.min(), df_coluna.max()
    if pd.isna(min_sec) or pd.isna(max_sec): return fig
    passo = 10 if (max_sec - min_sec) > 60 else 5
    ticks = list(range(int(min_sec) - passo, int(max_sec) + passo * 2, passo))
    ticktexts = [f"{t//60}:{t%60:02d}" for t in ticks]
    fig.update_yaxes(tickvals=ticks, ticktext=ticktexts, autorange="reversed")
    return fig

@st.cache_data
def processar_arquivos_completos(arquivos):
    sessoes_list, laps_list = [], []
    for arquivo in arquivos:
        try:
            fitfile = fitparse.FitFile(arquivo)
            for record in fitfile.get_messages('session'):
                dados = {data.name: data.value for data in record}
                dados['arquivo_id'] = arquivo.name
                sessoes_list.append(dados)
            for record in fitfile.get_messages('lap'):
                dados = {data.name: data.value for data in record}
                dados['arquivo_id'] = arquivo.name
                laps_list.append(dados)
        except: continue
            
    df_sessoes, df_laps = pd.DataFrame(), pd.DataFrame()
    if sessoes_list:
        df_s = pd.DataFrame(sessoes_list)
        df_sessoes['ID'], df_sessoes['Data'] = df_s.get('arquivo_id'), pd.to_datetime(df_s.get('start_time'))
        df_sessoes['Dist'] = pd.to_numeric(df_s.get('total_distance'), errors='coerce')
        df_sessoes['HR'] = pd.to_numeric(df_s.get('avg_heart_rate'), errors='coerce')
        speed = pd.to_numeric(df_s.get('avg_speed'), errors='coerce')
        df_sessoes['Pace_Sec'] = np.where(speed > 0, 100 / speed, np.nan)
        df_sessoes['Cadence'] = pd.to_numeric(df_s.get('avg_cadence'), errors='coerce')
        df_sessoes['Tem_GPS'] = df_s.get('start_position_lat').notna()
        df_sessoes = df_sessoes.dropna(subset=['Dist', 'Pace_Sec']).sort_values('Data')
        
    if laps_list:
        df_l = pd.DataFrame(laps_list)
        df_laps['ID'], df_laps['Data'] = df_l.get('arquivo_id'), pd.to_datetime(df_l.get('start_time'))
        df_laps['Dist'] = pd.to_numeric(df_l.get('total_distance'), errors='coerce')
        df_laps['Dist_Arredondada'] = (df_laps['Dist'] / 25).round() * 25
        df_laps['HR'] = pd.to_numeric(df_l.get('avg_heart_rate'), errors='coerce')
        df_laps['Cadence'] = pd.to_numeric(df_l.get('avg_cadence'), errors='coerce')
        df_laps['Tempo_Seg'] = pd.to_numeric(df_l.get('total_timer_time', df_l.get('total_elapsed_time')), errors='coerce')
        df_laps['Pace_Sec'] = np.where((df_laps['Dist'] > 0) & (df_laps['Tempo_Seg'] > 0), df_laps['Tempo_Seg'] / (df_laps['Dist'] / 100), np.nan)
        df_laps['ISH'] = df_laps['Pace_Sec'] / df_laps['HR']
        
        # Blindagem de braçadas (FILLNA)
        swolf_col = df_l.get('avg_swolf', pd.Series(np.nan, index=df_l.index))
        strokes_col = df_l.get('total_strokes', pd.Series(np.nan, index=df_l.index))
        
        swolf_garmin = pd.to_numeric(swolf_col, errors='coerce')
        strokes_calc = pd.to_numeric(strokes_col, errors='coerce').fillna(df_laps['Cadence'] * df_laps['Tempo_Seg'] / 60)
        swolf_calc = (df_laps['Pace_Sec'] / 4) + (strokes_calc / (df_laps['Dist'] / 25))
        df_laps['SWOLF'] = swolf_garmin.fillna(swolf_calc)
        df_laps['ICS'] = df_laps['SWOLF'] / df_laps['HR']
        df_laps = df_laps[df_laps['Dist'] >= 25].copy()
        
    return df_sessoes, df_laps

# --- MENU LATERAL ---
st.sidebar.title("🌊 Performance Liquids")

# Entrada da Chave API
chave_api_usuario = st.sidebar.text_input("🔑 Chave API Gemini:", type="password", help="Insira sua chave para ativar IA Multimodal.")

# --- ACESSO VIP (CLOSED BETA) ---
senha_vip = st.sidebar.text_input("🔐 Senha de Acesso (Beta):", type="password", help="Peça a senha ao treinador para liberar a Inteligência Artificial.")

modelo_ia = None
# A senha que você escolher para os seus amigos:
if senha_vip == "SWIM2026": 
    try:
        # Puxa a sua chave secretamente do cofre do Streamlit
        chave_secreta = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=chave_secreta)
        
        # Configura o motor da IA
        modelo_escolhido = "gemini-1.5-flash"
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods and 'flash' in m.name:
                modelo_escolhido = m.name.replace('models/', '') 
                break
        modelo_ia = genai.GenerativeModel(modelo_escolhido)
        st.sidebar.success("✨ Acesso IA Liberado!")
    except Exception as e:
        st.sidebar.error("Erro de conexão com o servidor. Tente mais tarde.")
elif senha_vip:
    st.sidebar.error("Senha incorreta.")

st.sidebar.markdown("---")
modo_avancado = st.sidebar.toggle("🔬 Modo Performance (Científico)", value=st.session_state.perfil_atleta.get('nivel') in ["Avançado", "Competitivo"])

opcoes_menu = ["👤 Perfil", "🏆 Tática de Guaratuba", "📊 Visão Geral", "🤖 Coach Virtual (IA)", "📚 Dicionário"]
if modo_avancado:
    opcoes_menu = ["👤 Perfil", "🏆 Tática de Guaratuba", "📊 Visão Geral", "🔬 Laboratório Fisiológico", "📚 Biblioteca de Séries", "🔍 Evolução de Fadiga", "📈 Evolução por Distância", "🤖 Coach Virtual (IA)", "📚 Dicionário"]

modulo = st.sidebar.radio("Navegação:", opcoes_menu)
st.sidebar.markdown("---")
arquivos_usuario = st.sidebar.file_uploader("Suba seus treinos (.FIT)", type=['fit'], accept_multiple_files=True)

# --- MÓDULOS ---

if modulo == "📚 Dicionário":
    st.title("Guia do Usuário")
    st.info("💡 **Glossário Tático:** Use este guia para traduzir números em performance.")
    st.markdown("""
    * **⏱️ Pace MM:SS:** Seu tempo para nadar 100m.
    * **❤️‍🔥 ISH (Speed-Heart):** Custo cardiovascular da velocidade. Se cair, você está mais eficiente.
    * **⚙️ ICS (Cardio-SWOLF):** Quebra técnica sob fadiga. Cruza SWOLF com Coração.
    """)

elif modulo == "👤 Perfil":
    st.title("Configuração do Atleta")
    st.info("💡 **Comece aqui:** Diga quem você é e sua meta. A IA calibrará os conselhos.")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.perfil_atleta['nome'] = st.text_input("Nome:", value=st.session_state.perfil_atleta.get('nome', ''))
        st.session_state.perfil_atleta['nivel'] = st.selectbox("Nível:", ["Iniciante", "Intermediário", "Avançado", "Competitivo"], index=2)
    with col2:
        st.session_state.perfil_atleta['objetivo'] = st.text_area("Objetivo:", value=st.session_state.perfil_atleta.get('objetivo', ''))
    if st.button("💾 Salvar Perfil"): st.success("Perfil atualizado!")

elif modulo == "🏆 Tática de Guaratuba":
    st.title("🏆 Alvo: Guaratuba (29/03)")
    st.info("💡 **O Ciclo Completo:** Use a Fase 1 antes da largada para estimar seu pace com IA. Use a Fase 2 no domingo à noite para subir o resultado oficial e auditar sua performance contra o pelotão de elite.")
    
    aba1, aba2 = st.tabs(["Fase 1: Pré-Prova (Tática e Estimativa)", "Fase 2: Pós-Prova (Autópsia de Resultados)"])
    
    with aba1:
        st.markdown("### 1. Parâmetros de Navegação")
        colA, colB = st.columns(2)
        with colA: temp_agua = st.number_input("Temp. Água (°C):", value=22.0)
        with colB: tide_type = st.selectbox("Maré na Largada:", ["Enchendo", "Vazando", "Preamar", "Baixamar"])
        
        colC, colD = st.columns(2)
        with colC: vento_vel = st.number_input("Vento (km/h):", value=15)
        with colD: vento_dir = st.selectbox("Direção do Vento (Origem):", ["Norte (N)", "Sul (S)", "Leste (E)", "Oeste (W)"])
        
        sentido_nado = st.text_input("🧭 Sentido Predominante (ex: Sul -> Norte):", placeholder="Ajuda a IA a orientar o mapa.")
        mapa_prova = st.file_uploader("📤 Subir Mapa do Percurso (Croqui):", type=['jpg', 'jpeg', 'png'])
        
        st.markdown("### 2. Projeção de Esforço")
        tempo_estimado = st.text_input("⏱️ Tempo Alvo Estimado (MM:SS):", placeholder="Ex: 25:00")
        
        tática_ia = st.button("🚀 Gerar Planejamento Tático com IA")
        
        if tática_ia:
            if modelo_ia and mapa_prova and sentido_nado and tempo_estimado:
                with st.spinner("Cruzando clima, mapa e estimativas..."):
                    from PIL import Image
                    imagem_pil = Image.open(mapa_prova)
                    prompt = f"""
                    Atleta de nível {st.session_state.perfil_atleta.get('nivel')}.
                    Clima: Água {temp_agua}°C, Maré {tide_type}, Vento {vento_vel}km/h {vento_dir}. Sentido: {sentido_nado}.
                    Tempo Alvo do Atleta: {tempo_estimado}.
                    Analise a imagem anexa (mapa). Gere (Markdown):
                    1. Navegação: Sugestão visual de mira nas boias.
                    2. Tática de Maré/Vento.
                    3. Viabilidade do Pace: O tempo alvo de {tempo_estimado} é aderente às condições de arrasto deste mar?
                    """
                    try:
                        resp = modelo_ia.generate_content([prompt, imagem_pil])
                        st.markdown(resp.text)
                    except Exception as e: st.error(f"Erro IA: {e}")
            else:
                st.warning("⚠️ Insira a Chave API, o Mapa, o Sentido e o Tempo Alvo para uma análise completa.")

    with aba2:
        st.markdown("### 1. A Verdade do Cronômetro")
        tempo_real = st.text_input("⏱️ Tempo Real Executado Oficial (MM:SS):", placeholder="Ex: 24:11")
        
        st.markdown("### 2. O Raio-X do Pelotão")
        arquivo_resultados = st.file_uploader("📤 Subir Resultados Oficiais (TXT, CSV ou PDF leve):", type=['txt', 'csv', 'pdf'], help="Suba a lista de classificação da organização para a IA encontrar o pelotão.")
        
        analise_pos_prova = st.button("🔬 Iniciar Auditoria de Desempenho (IA)")
        
        if analise_pos_prova:
            if modelo_ia and arquivo_resultados and tempo_real:
                with st.spinner("Analisando o pelotão de elite e a sua posição..."):
                    # Lê o arquivo dependendo do formato
                    conteudo_arquivo = ""
                    try:
                        if arquivo_resultados.name.endswith('.csv'):
                            df_res = pd.read_csv(arquivo_resultados)
                            conteudo_arquivo = df_res.head(50).to_string() # Manda os top 50 para a IA ter base
                        elif arquivo_resultados.name.endswith('.txt'):
                            conteudo_arquivo = arquivo_resultados.getvalue().decode("utf-8")[:3000]
                        else:
                            st.warning("⚠️ Suporte completo a PDFs pesados será adicionado na V2. Usando dados básicos por enquanto.")
                            conteudo_arquivo = "Arquivo PDF enviado."
                            
                        prompt_pos = f"""
                        Atleta: {st.session_state.perfil_atleta.get('nome', 'O Atleta')}.
                        Tempo Estimado Pré-Prova: {tempo_estimado if 'tempo_estimado' in locals() else 'N/A'}.
                        Tempo Real Executado: {tempo_real}.
                        
                        Aqui estão os resultados oficiais da prova (Top colocados):
                        {conteudo_arquivo}
                        
                        Aja como um Cientista de Dados Esportivos. Entregue:
                        1. **Análise de Pace (Real vs Estimado):** Quão preciso foi o atleta?
                        2. **Gap para a Elite:** Calcule a diferença de tempo e de pace do atleta para o líder geral (pelotão de elite).
                        3. **Foco do Próximo Macrociclo:** Baseado nesse gap, o que ele precisa treinar para encostar na ponta da categoria dele no próximo evento?
                        """
                        resp_pos = modelo_ia.generate_content(prompt_pos)
                        st.markdown(resp_pos.text)
                    except Exception as e:
                        st.error(f"Erro ao ler resultados: {e}")
            else:
                st.warning("⚠️ Insira o Tempo Real e suba o arquivo de Resultados Oficiais para a auditoria.")

elif arquivos_usuario:
    # ... (DEMAIS MÓDULOS INTACTOS, SEM ALTERAÇÃO)
    df_sess, df_laps = processar_arquivos_completos(arquivos_usuario)
    if not df_sess.empty:
        df_sess['Label'] = df_sess['Data'].dt.strftime('%d/%m/%Y') + " (" + df_sess['Dist'].astype(int).astype(str) + "m)"
        opcoes_treino = dict(zip(df_sess['Label'], df_sess['ID']))

        if modulo == "📊 Visão Geral":
            st.title("Dashboard de Desempenho")
            st.info("💡 **Termômetro Diário:** Setas comparam hoje com média dos últimos 30 dias.")
            data_recente = df_sess['Data'].max()
            df_ultimo = df_sess.iloc[-1:]
            df_mes = df_sess[df_sess['Data'] >= (data_recente - pd.Timedelta(days=30))]
            laps_ultimo = df_laps[df_laps['ID'] == df_ultimo['ID'].values[0]]
            laps_mes = df_laps[df_laps['ID'].isin(df_mes['ID'])]
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Hoje**")
                st.metric("Pace Médio", format_pace(laps_ultimo['Pace_Sec'].mean()))
                st.metric("FC Média", f"{laps_ultimo['HR'].mean():.0f} bpm")
                st.metric("SWOLF", f"{laps_ultimo['SWOLF'].mean():.1f}")
            with c2:
                st.markdown("**Últimos 30 Dias**")
                p_mes = laps_mes['Pace_Sec'].mean()
                d_p = laps_ultimo['Pace_Sec'].mean() - p_mes
                st.metric("Pace", format_pace(p_mes), delta=f"{d_p:.1f}s", delta_color="inverse")
                hr_mes = laps_mes['HR'].mean()
                d_h = laps_ultimo['HR'].mean() - hr_mes
                st.metric("FC", f"{hr_mes:.0f} bpm", delta=f"{d_h:.1f} bpm", delta_color="inverse")
            with c3:
                st.markdown("**Histórico Global**")
                st.metric("Volume total", f"{df_sess['Dist'].sum() / 1000:.1f} km")
                if modo_avancado:
                    st.metric("ICS Médio", f"{df_laps['ICS'].mean():.3f}")

        elif modulo == "🔬 Laboratório Fisiológico" and modo_avancado:
            st.title("Laboratório Biomecânico")
            aba1, aba2, aba3, aba4 = st.tabs(["Custo Motor", "Pacing", "Pool/OW", "Eficiência ICS"])
            with aba1:
                fig_scatter = px.scatter(df_sess, x="HR", y="Pace_Sec", color="Dist", title="Pace vs Custo Cardíaco")
                ticks = [100, 110, 120, 130]
                text = ["1:40", "1:50", "2:00", "2:10"]
                fig_scatter.update_yaxes(tickvals=ticks, ticktext=text, autorange="reversed")
                st.plotly_chart(fig_scatter, use_container_width=True)
            with aba3:
                st.markdown("### Comparador Tático: Piscina x Águas Abertas")
                df_ow = df_sess.groupby(['Tem_GPS']).agg(Pace_Medio=('Pace_Sec', 'mean'), Cadencia_Media=('Cadence', 'mean')).reset_index()
                df_ow['Modo'] = np.where(df_ow['Tem_GPS'], 'OW (GPS)', 'Pool (No GPS)')
                c1, c2 = st.columns(2)
                c1.plotly_chart(px.bar(df_ow, x='Modo', y='Pace_Medio', title='Pace (s/100m)'), use_container_width=True)
                c2.plotly_chart(px.bar(df_ow, x='Modo', y='Cadencia_Media', title='Cadência (spm)'), use_container_width=True)

        elif modulo == "📈 Evolução por Distância" and modo_avancado:
            st.title("Curva de Performance")
            dist_disp = sorted([int(d) for d in df_laps['Dist_Arredondada'].dropna().unique() if 25 <= d <= 5000])
            if dist_disp:
                dist_alvo = st.selectbox("🎯 Distância:", dist_disp)
                l_filt = df_laps[df_laps['Dist_Arredondada'] == dist_alvo]
                media_dia = l_filt.groupby(l_filt['Data'].dt.date).agg(Pace_Medio=('Pace_Sec', 'mean'), Qtd=('Dist', 'count')).reset_index()
                media_dia['Tendencia'] = media_dia['Pace_Medio'].rolling(window=3, min_periods=1).mean()
                fig_trend = px.scatter(media_dia, x='Data', y='Pace_Medio', size='Qtd', title=f"Tiros de {dist_alvo}m")
                fig_trend.add_trace(go.Scatter(x=media_dia['Data'], y=media_dia['Tendencia'], mode='lines', name='Curva', line=dict(color='red', width=3)))
                fig_trend = formatar_eixo_y_pace_absoluto(fig_trend, media_dia['Pace_Medio'])
                st.plotly_chart(fig_trend, use_container_width=True)

        elif modulo == "📚 Biblioteca de Séries" and modo_avancado:
            st.title("Extrair Séries")
            opcoes_disponiveis = {k: v for k, v in opcoes_treino.items() if v not in st.session_state.treinos_usados}
            if opcoes_disponiveis:
                treino_sel = st.selectbox("Treino:", list(opcoes_disponiveis.keys()))
                laps_treino = df_laps[df_laps['ID'] == opcoes_disponiveis[treino_sel]].reset_index(drop=True)
                laps_treino['Tiro Nº'] = laps_treino.index + 1
                df_ex = laps_treino[['Tiro Nº', 'Dist', 'Pace_Sec', 'HR']].copy()
                df_ex.insert(0, "Incluir", False)
                edited_df = st.data_editor(df_ex[['Incluir', 'Tiro Nº', 'Dist', 'Pace_Sec', 'HR']], hide_index=True)
                laps_sel = laps_treino[edited_df["Incluir"]].copy()
                if not laps_sel.empty:
                    if st.button("💾 Salvar"):
                        laps_sel['Tiro Nº'] = range(1, len(laps_sel) + 1)
                        st.session_state.biblioteca_series[f"Série ({treino_sel[:10]})"] = laps_sel
                        st.session_state.treinos_usados.add(opcoes_disponiveis[treino_sel])
                        st.rerun()

        elif modulo == "🤖 Coach Virtual (IA)":
            st.title("🧠 Diário NLP")
            st.info("💡 **Sentimento (Práxis):** Relate o treino e a IA analisará confiança e motivação.")
            treino_alvo = st.selectbox("Selecione treino:", list(opcoes_treino.keys()))
            dados = df_sess[df_sess['ID'] == opcoes_treino[treino_alvo]].iloc[0]
            contexto_ia = f"Volume: {dados['Dist']}m | Pace: {format_pace(dados.get('Pace_Sec'))}/100m"
            relato_atleta = st.text_area("Diário de Bordo Performance Liquids:")
            if st.button("🚀 Diagnóstico NLP"):
                if modelo_ia:
                    with st.spinner("Lendo..."):
                        prompt = f"Coach Natação nível {st.session_state.perfil_atleta.get('nivel')}. Dados: {contexto_ia}. Relato: {relato_atleta}. Entregue: Diagnóstico Fisiológico, Análise NLP Sentimento, Score Mental 0-10."
                        resp = modelo_ia.generate_content(prompt)
                        st.markdown(resp.text)
                else: st.warning("⚠️ Insira Chave API.")

    else: st.error("Falha ao ler dados.")
else: st.write("👈 *Aguardando FIT para ignição...*")
