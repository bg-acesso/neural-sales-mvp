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

app = Flask(__name__)

@app.route('/')
def home():
    return "Neural Sales Ops is Active", 200

# --- CONFIGURAÇÃO ---
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

def process_file(file_path):
    try:
        print(f"📥 Baixando arquivo: {file_path}...", flush=True)
        file_data = supabase.storage.from_(BUCKET_IN).download(file_path)
        text_content = file_data.decode('utf-8')
        
        print(f"🧠 Analisando com DeepSeek: {file_path}...", flush=True)
        prompt = f"Analise esta conversa de vendas e sugira a próxima ação:\n\n{text_content}"
        response = llm.invoke([SystemMessage(content="Sales Coach."), HumanMessage(content=prompt)])
        
        # Nome do relatório (substitui barras para não criar subpastas no bucket de saída se não quiser)
        report_name = f"RELATORIO_{file_path.replace('/', '_')}_{int(time.time())}.md"
        
        print(f"📤 Enviando relatório: {report_name}...", flush=True)
        supabase.storage.from_(BUCKET_OUT).upload(
            path=report_name,
            file=response.content.encode('utf-8'),
            file_options={"content-type": "text/markdown"}
        )
        
        print(f"🗑 Deletando entrada: {file_path}...", flush=True)
        supabase.storage.from_(BUCKET_IN).remove([file_path])
        return True
    except Exception as e:
        print(f"❌ Erro ao processar {file_path}: {e}", flush=True)
        return False

def worker_loop():
    print("🚀 WORKER INICIADO: Monitorando Buckets...", flush=True)
    vendedores = ["Vendedor_Ana", "Vendedor_Bruno", "Vendedor_Carlos"]
    
    while True:
        try:
            for vendedor in vendedores:
                # Log de batimento cardíaco para sabermos que o bot não travou
                print(f"🧐 Verificando pasta: {vendedor}...", flush=True)
                
                res = supabase.storage.from_(BUCKET_IN).list(vendedor)
                
                if res:
                    for f in res:
                        fname = f['name']
                        # Ignora arquivos do sistema ou pastas vazias
                        if fname.endswith('.txt'):
                            # Se o nome já tiver a pasta, não duplica. Senão, monta o path.
                            full_path = fname if "/" in fname else f"{vendedor}/{fname}"
                            print(f"🎯 Arquivo encontrado! Alvo: {full_path}", flush=True)
                            process_file(full_path)
            
            # Espera 30 segundos antes da próxima ronda
            time.sleep(30)
            
        except Exception as e:
            print(f"⚠️ Erro no Loop: {e}", flush=True)
            time.sleep(30)

# Inicia o robô em background
t = Thread(target=worker_loop, daemon=True)
t.start()
print("🤖 Thread de monitoramento disparada.", flush=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)