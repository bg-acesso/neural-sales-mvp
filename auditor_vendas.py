import os
import time
import hashlib
from dotenv import load_dotenv

# Supabase Imports
from supabase import create_client, Client

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# --- CONFIGURAÇÃO ---
INPUT_ROOT = "inputs"
OUTPUT_ROOT = "outputs"

# Config Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Config DeepSeek
llm = ChatOpenAI(
    model='deepseek-chat', 
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base='https://api.deepseek.com',
    temperature=0.2
)

# --- 1. GESTÃO DE BANCO DE DADOS (SUPABASE) ---

def get_file_state(file_path):
    """Busca o estado do arquivo no Supabase"""
    try:
        response = supabase.table("sales_memory") \
            .select("last_hash, last_summary") \
            .eq("file_path", file_path) \
            .execute()
        
        # Se houver dados, retorna o primeiro registro
        if response.data and len(response.data) > 0:
            return response.data[0]['last_hash'], response.data[0]['last_summary']
        return None, None
    except Exception as e:
        print(f"⚠ Erro ao conectar Supabase: {e}")
        return None, None

def update_file_state(file_path, salesperson, current_hash, summary):
    """Upsert (Insere ou Atualiza) no Supabase"""
    data = {
        "file_path": file_path,
        "salesperson": salesperson,
        "last_hash": current_hash,
        "last_summary": summary,
        "updated_at": "now()"
    }
    
    try:
        # on_conflict garante que atualiza se o file_path já existir
        supabase.table("sales_memory").upsert(data, on_conflict="file_path").execute()
    except Exception as e:
        print(f"⚠ Erro ao salvar no Supabase: {e}")

# --- 2. FUNÇÕES AUXILIARES ---
def calculate_file_hash(filepath):
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def read_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

# --- 3. INTELIGÊNCIA ---

def analyze_update(salesperson_name, filename, full_text, previous_summary):
    print(f"   ☁️  Auditor (Cloud) analisando {salesperson_name}...")
    
    context_prompt = ""
    if previous_summary:
        context_prompt = f"O QUE JÁ SABEMOS:\n{previous_summary}\n\n"
    
    prompt = f"""
    {context_prompt}
    VENDEDOR: {salesperson_name}
    ARQUIVO: {filename}
    
    Analise a continuação da conversa.
    1. O cliente deu sinais de compra? (Score 0-100)
    2. O vendedor cometeu algum erro FATAL nas novas mensagens?
    3. Qual a próxima mensagem EXATA que deve ser enviada?
    
    Responda no formato:
    [RELATORIO]
    (Feedback direto e tático)
    [RESUMO]
    (Resumo técnico atualizado)
    
    CONVERSA ATUALIZADA:
    {full_text}
    """
    
    try:
        response = llm.invoke([SystemMessage(content="Sales Ops AI."), HumanMessage(content=prompt)])
        content = response.content
        
        parts = content.split("[RESUMO]")
        report = parts[0].replace("[RELATORIO]", "").strip()
        new_summary = parts[1].strip() if len(parts) > 1 else "Resumo mantido."
        
        return report, new_summary
    except Exception as e:
        print(f"❌ Erro na API da DeepSeek: {e}")
        return "Erro na análise.", previous_summary

# --- 4. LOOP PRINCIPAL ---
def start_watchdog():
    print(f"🚀 NEURAL SALES OPS CONECTADO AO SUPABASE.")
    print(f"👀 Monitorando '{INPUT_ROOT}'...")
    
    # Cria pastas locais se não existirem (apenas para garantir)
    os.makedirs(INPUT_ROOT, exist_ok=True)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    try:
        while True:
            for root, dirs, files in os.walk(INPUT_ROOT):
                current_folder_name = os.path.basename(root)
                if current_folder_name == INPUT_ROOT: continue

                for filename in files:
                    if filename.endswith(".txt"):
                        full_path = os.path.join(root, filename)
                        relative_path = os.path.relpath(full_path, INPUT_ROOT)
                        
                        # 1. Checagem Local (Hash)
                        current_hash = calculate_file_hash(full_path)
                        
                        # 2. Checagem Remota (Supabase)
                        last_hash, last_summary = get_file_state(relative_path)
                        
                        if current_hash != last_hash:
                            if last_hash is None:
                                print(f"\n🆕 [SUPABASE] Novo registro: {filename}")
                            else:
                                print(f"\n🔄 [SUPABASE] Atualização: {filename}")
                            
                            text = read_file(full_path)
                            report, new_summary = analyze_update(current_folder_name, filename, text, last_summary)
                            
                            # Salvar output localmente (para entregar ao cliente)
                            output_dir = os.path.join(OUTPUT_ROOT, current_folder_name)
                            os.makedirs(output_dir, exist_ok=True)
                            timestamp = int(time.time())
                            report_filename = f"AUDITORIA_{filename.replace('.txt', '')}_{timestamp}.md"
                            
                            with open(os.path.join(output_dir, report_filename), "w", encoding="utf-8") as f:
                                f.write(report)
                                
                            # Atualizar nuvem
                            update_file_state(relative_path, current_folder_name, current_hash, new_summary)
                            print(f"✅ Dados sincronizados com a nuvem.")
            
            time.sleep(10)

    except KeyboardInterrupt:
        print("\n🛑 Desconectando...")

if __name__ == "__main__":
    start_watchdog()