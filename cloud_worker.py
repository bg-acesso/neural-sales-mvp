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
    return "Neural Sales Ops is 100% Operational", 200

# --- CONFIGURAÇÃO ---
url = os.getenv("SUPABASE_URL")
if url and not url.endswith('/'): url += '/'
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
        # Extrai o nome do vendedor da pasta (ex: Vendedor_Ana/chat.txt -> Vendedor_Ana)
        salesperson = file_path.split('/')[0] if '/' in file_path else "Desconhecido"
        
        print(f"📥 Processando {file_path} de {salesperson}...", flush=True)
        file_data = supabase.storage.from_(BUCKET_IN).download(file_path)
        text_content = file_data.decode('utf-8')
        
        # 1. PEGAR MEMÓRIA ANTERIOR (Opcional para o prompt)
        # Vamos apenas gerar e salvar para este MVP
        
        print(f"🧠 Consultando DeepSeek...", flush=True)
        prompt = f"""
        Vendedor: {salesperson}
        Analise esta conversa e responda em duas partes:
        [RELATORIO]
        (Dicas táticas para o vendedor)
        [RESUMO]
        (Resumo técnico de 1 parágrafo do estado da venda para o banco de dados)
        
        CONVERSA:
        {text_content}
        """
        
        response = llm.invoke([SystemMessage(content="Sales Ops Director."), HumanMessage(content=prompt)])
        full_content = response.content
        
        # Separa Relatório de Resumo
        parts = full_content.split("[RESUMO]")
        report = parts[0].replace("[RELATORIO]", "").strip()
        summary = parts[1].strip() if len(parts) > 1 else "Sem resumo."

        # 2. SALVAR RELATÓRIO NO STORAGE
        timestamp = int(time.time())
        report_name = f"RELATORIO_{file_path.replace('/', '_')}_{timestamp}.md"
        supabase.storage.from_(BUCKET_OUT).upload(
            path=report_name,
            file=report.encode('utf-8'),
            file_options={"content-type": "text/markdown"}
        )

        # 3. ATUALIZAR TABLE EDITOR (MEMÓRIA)
        print(f"💾 Salvando memória na Tabela...", flush=True)
        db_data = {
            "file_path": file_path,
            "salesperson": salesperson,
            "last_summary": summary,
            "updated_at": "now()"
        }
        # Upsert: Insere novo ou atualiza se o file_path já existir
        supabase.table("sales_memory").upsert(db_data, on_conflict="file_path").execute()
        
        # 4. LIMPAR ENTRADA
        print(f"🗑 Deletando entrada e finalizando.", flush=True)
        supabase.storage.from_(BUCKET_IN).remove([file_path])
        
        return True
    except Exception as e:
        print(f"❌ Erro crítico: {e}", flush=True)
        return False

def worker_loop():
    print("🚀 WORKER OPERACIONAL: Escaneando Nuvem...", flush=True)
    while True:
        try:
            root_files = supabase.storage.from_(BUCKET_IN).list()
            if root_files:
                for item in root_files:
                    name = item['name']
                    if name.startswith("Vendedor_"):
                        sub_files = supabase.storage.from_(BUCKET_IN).list(name)
                        if sub_files:
                            for sf in sub_files:
                                if sf['name'].endswith('.txt'):
                                    full_path = f"{name}/{sf['name']}"
                                    process_file(full_path)
            time.sleep(25)
        except Exception as e:
            print(f"⚠️ Erro Loop: {e}", flush=True)
            time.sleep(30)

t = Thread(target=worker_loop, daemon=True)
t.start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)