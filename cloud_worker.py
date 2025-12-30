import os
import time
import hashlib
from threading import Thread
from flask import Flask
from dotenv import load_dotenv

# Supabase & LangChain
from supabase import create_client, Client
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# --- CONFIGURAÇÃO WEB (PARA O RENDER NÃO DERRUBAR) ---
app = Flask(__name__)

@app.route('/')
def home():
    return "Neural Sales Ops is Running 24/7."

# --- CONFIGURAÇÃO DO SISTEMA ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

llm = ChatOpenAI(
    model='deepseek-chat', 
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base='https://api.deepseek.com',
    temperature=0.2
)

BUCKET_IN = "sales-logs"
BUCKET_OUT = "sales-reports"

# --- LÓGICA DO NEGÓCIO ---

def process_file(file_object):
    filename = file_object['name'] # ex: Vendedor_Ana/cliente1.txt
    print(f"📥 Baixando e processando: {filename}...")
    
    # 1. Baixar conteúdo do arquivo do Storage
    file_data = supabase.storage.from_(BUCKET_IN).download(filename)
    text_content = file_data.decode('utf-8')
    
    # 2. Analisar (DeepSeek)
    prompt = f"""
    Analise esta conversa de vendas.
    Arquivo: {filename}
    
    1. Resumo da situação.
    2. Erros críticos.
    3. Ação recomendada (Script de resposta).
    
    CONVERSA:
    {text_content}
    """
    
    response = llm.invoke([SystemMessage(content="Sales Ops AI."), HumanMessage(content=prompt)])
    report_content = response.content
    
    # 3. Salvar Relatório no Storage de Saída
    timestamp = int(time.time())
    report_name = f"AUDITORIA_{filename.replace('.txt', '')}_{timestamp}.md"
    
    # Precisamos garantir que a pasta do vendedor exista no nome do arquivo ou path
    # O Supabase aceita paths no nome ex: "Vendedor_Ana/Relatorio.md"
    
    print(f"📤 Uploading relatório: {report_name}")
    supabase.storage.from_(BUCKET_OUT).upload(
        path=report_name,
        file=report_content.encode('utf-8'),
        file_options={"content-type": "text/markdown"}
    )
    
    # 4. Mover arquivo processado para uma pasta "processed" (Opcional - aqui vamos apenas deletar para não processar de novo no MVP)
    # No MVP, vamos deletar o arquivo de entrada para não gastar API processando ele 1000 vezes em loop
    print(f"🗑 Deletando entrada processada: {filename}")
    supabase.storage.from_(BUCKET_IN).remove([filename])

def worker_loop():
    print("🚀 WORKER INICIADO: Monitorando Supabase Storage...")
    while True:
        try:
            # Lista arquivos no bucket de entrada
            # Tenta listar na raiz e subpastas
            files = supabase.storage.from_(BUCKET_IN).list()
            
            for f in files:
                if f['name'].endswith('.txt'):
                    process_file(f)
                    
            # Dorme 30 segundos para economizar
            time.sleep(30)
            
        except Exception as e:
            print(f"⚠ Erro no loop: {e}")
            time.sleep(30)

# --- INICIALIZAÇÃO HÍBRIDA ---
def run():
    # Inicia o worker em background
    t = Thread(target=worker_loop)
    t.start()
    
    # Inicia o servidor web (Bloqueante)
    # Render vai procurar a porta via variável de ambiente PORT ou padrão 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

if __name__ == "__main__":
    run()