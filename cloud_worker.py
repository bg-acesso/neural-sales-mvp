import os
import time
from threading import Thread
from flask import Flask
from dotenv import load_dotenv
from supabase import create_client, Client
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()
app = Flask(__name__)

@app.route('/')
def home():
    return "Neural Sales Ops is Active", 200

# --- CONFIGURAÇÃO ---
# Corrigindo o erro do "trailing slash" automaticamente
url = os.getenv("SUPABASE_URL")
if url and not url.endswith('/'):
    url += '/'

SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, SUPABASE_KEY)

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
        print(f"📥 Baixando: {file_path}...", flush=True)
        file_data = supabase.storage.from_(BUCKET_IN).download(file_path)
        text_content = file_data.decode('utf-8')
        
        print(f"🧠 Analisando: {file_path}...", flush=True)
        prompt = f"Analise esta conversa de vendas e sugira a próxima ação comercial:\n\n{text_content}"
        response = llm.invoke([SystemMessage(content="Sales Coach."), HumanMessage(content=prompt)])
        
        # Limpando o nome do arquivo para o relatório
        clean_name = file_path.replace('/', '_').replace('\\', '_')
        report_name = f"RELATORIO_{clean_name}_{int(time.time())}.md"
        
        print(f"📤 Enviando relatório: {report_name}...", flush=True)
        supabase.storage.from_(BUCKET_OUT).upload(
            path=report_name,
            file=response.content.encode('utf-8'),
            file_options={"content-type": "text/markdown"}
        )
        
        print(f"🗑 Deletando entrada: {file_path}...", flush=True)
        supabase.storage.from_(BUCKET_IN).remove([file_path])
        print(f"✅ Sucesso total: {file_path}", flush=True)
        return True
    except Exception as e:
        print(f"❌ Erro ao processar {file_path}: {e}", flush=True)
        return False

def worker_loop():
    print("🚀 WORKER INICIADO: Monitoramento Total...", flush=True)
    vendedores = ["Vendedor_Ana", "Vendedor_Bruno", "Vendedor_Carlos"]
    
    while True:
        try:
            for vendedor in vendedores:
                # Tentativa de listar a pasta
                res = supabase.storage.from_(BUCKET_IN).list(vendedor)
                
                # Log de debug para sabermos exatamente o que o Supabase está respondendo
                if res:
                    for f in res:
                        fname = f['name']
                        # Ignora arquivos vazios de sistema (.placeholder)
                        if fname.endswith('.txt'):
                            # RECONSTRUÇÃO DO CAMINHO:
                            # O Supabase às vezes retorna apenas o nome, às vezes o path completo.
                            if "/" in fname or "\\" in fname:
                                full_path = fname.replace('\\', '/')
                            else:
                                full_path = f"{vendedor}/{fname}"
                            
                            print(f"🎯 ALVO IDENTIFICADO: {full_path}", flush=True)
                            process_file(full_path)
                else:
                    # Se não encontrar nada na pasta específica, tentamos o 'root' como último recurso
                    pass
            
            time.sleep(20) # Checagem mais rápida (20s)
            
        except Exception as e:
            print(f"⚠️ Erro no Loop: {e}", flush=True)
            time.sleep(30)

t = Thread(target=worker_loop, daemon=True)
t.start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)