"""
Chatbot com Gemini AI e Streamlit

Funcionalidades:
- Interface de chat limpa e responsiva
- Configuração de API KEY na sidebar
- Histórico de conversa persistente
- Botão para limpar o histórico
- Tratamento de erros robusto
- Respostas em tempo real (streaming)
"""

import google.generativeai as gemini
import streamlit as st
from typing import List, Dict

# ============================================
# CONSTANTES E CONFIGURAÇÕES
# ============================================
DEFAULT_MESSAGE = {
    "role": "assistant", 
    "content": "Olá! Eu sou o Gemini. Como posso ajudar você hoje?"
}
MODEL_NAME = "gemini-1.5-flash"  # Modelo mais recente
SAFETY_SETTINGS = {
    'HARM_CATEGORY_HARASSMENT': 'BLOCK_ONLY_HIGH',
    'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_ONLY_HIGH',
    'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_ONLY_HIGH',
    'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_ONLY_HIGH',
}

# ============================================
# FUNÇÕES DE INICIALIZAÇÃO
# ============================================
def initialize_session_state():
    """Inicializa o histórico de mensagens se não existir"""
    if "messages" not in st.session_state:
        st.session_state.messages = [DEFAULT_MESSAGE]

def configure_gemini(api_key: str):
    """Configura a API do Gemini"""
    try:
        gemini.configure(api_key=api_key)
    except Exception as e:
        st.error(f"Erro na configuração da API: {str(e)}")
        st.stop()

# ============================================
# COMPONENTES DA INTERFACE
# ============================================
def setup_sidebar():
    """Configura a barra lateral com inputs e controles"""
    with st.sidebar:
        st.title("⚙️ Configurações")
        st.markdown("---")
        
        # Input da API Key
        api_key = st.text_input(
            "🔑 Insira sua API Key do Gemini",
            type="password",
            help="Obtenha sua API Key em: https://aistudio.google.com/app/apikey"
        )
        
        if not api_key:
            st.warning("Por favor, insira sua API Key para continuar")
            st.stop()
        
        configure_gemini(api_key)
        
        # Controles adicionais
        st.markdown("---")
        st.markdown("### 🛠️ Controles")
        
        if st.button("🧹 Limpar Histórico", help="Reinicia a conversa"):
            st.session_state.messages = [DEFAULT_MESSAGE]
            st.rerun()
        
        return api_key

def display_chat_history():
    """Exibe o histórico completo da conversa"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# ============================================
# LÓGICA DO CHATBOT
# ============================================
def generate_gemini_response(prompt: str) -> str:
    """Gera resposta usando a API do Gemini com streaming"""
    try:
        model = gemini.GenerativeModel(MODEL_NAME)
        
        # Prepara o histórico como contexto
        chat_history = [
            {"role": msg["role"], "parts": [msg["content"]]} 
            for msg in st.session_state.messages
        ]
        
        # Cria o stream de resposta
        response = model.generate_content(
            contents=chat_history + [{"role": "user", "parts": [prompt]}],
            safety_settings=SAFETY_SETTINGS,
            stream=True
        )
        
        return response
        
    except Exception as e:
        raise Exception(f"Erro ao gerar resposta: {str(e)}")

def process_user_input(prompt: str):
    """Processa o input do usuário e gera a resposta"""
    # Adiciona mensagem do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Exibe mensagem do usuário
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Gera e exibe resposta do assistente
    with st.chat_message("assistant"):
        try:
            message_placeholder = st.empty()
            full_response = ""
            
            with st.spinner("🔍 Pensando..."):
                response = generate_gemini_response(prompt)
                
                # Stream da resposta
                for chunk in response:
                    if chunk.text:
                        full_response += chunk.text
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
            
            # Adiciona resposta ao histórico
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
            
        except Exception as e:
            error_msg = f"⚠️ Ocorreu um erro: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )

# ============================================
# FUNÇÃO PRINCIPAL
# ============================================
def main():
    """Função principal do aplicativo"""
    # Configuração da página
    st.set_page_config(
        page_title="Chatbot Gemini",
        page_icon="🤖",
        layout="centered"
    )
    
    st.title("🤖 Chatbot com Gemini AI")
    st.caption("Converse com a IA generativa do Google")
    
    # Configura sidebar e obtém API Key
    setup_sidebar()
    
    # Inicializa estado da sessão
    initialize_session_state()
    
    # Exibe histórico do chat
    display_chat_history()
    
    # Input do usuário
    if prompt := st.chat_input("Digite sua mensagem..."):
        process_user_input(prompt)

if __name__ == "__main__":
    main()