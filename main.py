import os
import time
from typing import TypedDict, Optional
from datetime import datetime
from dotenv import load_dotenv

# LangChain / LangGraph imports
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# Carrega variáveis de ambiente (.env)
load_dotenv()

# --- CONFIGURAÇÃO ---
class Config:
    # Certifique-se de ter GROQ_API_KEY no seu arquivo .env ou defina aqui
    GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
    MODEL_NAME = "llama-3.3-70b-versatile"

# Verifica se a API Key existe
if not Config.GROQ_API_KEY:
    raise ValueError("A chave da API Groq não foi encontrada. Defina GROQ_API_KEY no .env")

# --- 1. DEFINIÇÃO DO ESTADO ---
class RefinariaState(TypedDict):
    input_text: str
    estrutura: Optional[str]
    post: Optional[str]
    image_prompt: Optional[str]
    image_path: Optional[str]
    approved: Optional[bool]
    timestamp: Optional[str]

# --- 2. SETUP FERRAMENTAS (MOCKS E UTILS) ---

# Simulação da ferramenta de imagem (Flux)
class FluxImageTool:
    def run(self, prompt: str) -> str:
        print(f"\n🎨 [Flux] Gerando imagem para: '{prompt}'...")
        # Simulação de delay de API
        time.sleep(2) 
        # Aqui você colocaria a chamada real para a API (Replicate, HuggingFace, etc)
        fake_path = f"imagem_gerada_{int(time.time())}.png"
        print(f"🎨 [Flux] Imagem salva em: {fake_path} (Simulação)")
        return fake_path

# Função para salvar o arquivo final
def salvar_resultado(resultado, tarefas, entrada_usuario, timestamp):
    filename = f"post_linkedin_{timestamp}.md"
    content = f"""# Post LinkedIn Gerado em {timestamp}

## Entrada Original
{entrada_usuario}

## --- CONTEÚDO FINAL ---
{resultado if resultado else "Conteúdo salvo via estado."}
"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n💾 Arquivo salvo com sucesso: {filename}")

# Instancia ferramentas
flux_tool = FluxImageTool()
llm = ChatGroq(
    api_key=Config.GROQ_API_KEY,
    model=Config.MODEL_NAME,
    temperature=0.7
)

# --- 3. NÓS DO GRAFO ---

def analisar_estrutura(state: RefinariaState) -> RefinariaState:
    print("\n🕵️  [1/4] Analisando estrutura...")
    prompt = f"""
    Analise este conteúdo bruto e defina a melhor estrutura para um post de LinkedIn:

    {state['input_text']}

    Identifique:
    1. Gancho
    2. Problema
    3. Solução
    4. CTA
    5. Tom
    """
    resp = llm.invoke(prompt)
    return {"estrutura": resp.content}

def gerar_post(state: RefinariaState) -> RefinariaState:
    print("\n✍️  [2/4] Escrevendo post...")
    prompt = f"""
    Usando a estrutura abaixo, escreva um post para LinkedIn.

    REGRAS:
    - Até 1300 caracteres
    - Gancho forte
    - Parágrafos curtos
    - 3–5 hashtags no final
    - Final com pergunta ou CTA

    ESTRUTURA:
    {state['estrutura']}
    """
    resp = llm.invoke(prompt)
    return {"post": resp.content}

def gerar_prompt_imagem(state: RefinariaState) -> RefinariaState:
    print("\n🎨 [3/4] Criando prompt visual...")
    prompt = f"""
    Create an image prompt in English (max 15 words) representing this LinkedIn post.

    POST:
    {state['post']}

    Rules:
    - No text
    - No faces
    - Abstract, corporate, modern
    """
    resp = llm.invoke(prompt)
    return {"image_prompt": resp.content.strip()}

def gerar_imagem(state: RefinariaState) -> RefinariaState:
    print("\n🖼️  [4/4] Gerando imagem...")
    # Chama a tool instanciada
    path = flux_tool.run(state["image_prompt"])
    return {"image_path": path}

# Nó de revisão (Humano no loop)
# Nota: No LangGraph, é comum usar uma função condicional para inputs humanos,
# ou um nó que apenas coleta dados. Aqui, usaremos como função de roteamento.
def revisar_input(state: RefinariaState) -> str:
    print("\n" + "="*40)
    print("📄 PREVIEW DO POST")
    print("="*40)
    print(state["post"])
    print("-" * 20)
    print(f"PROMPT IMAGEM: {state['image_prompt']}")
    print("="*40)
    
    resp = input("\nSalvar este conteúdo? (s/n): ").lower()
    if resp == "s":
        return "salvar"
    return "abortar"

def salvar(state: RefinariaState) -> RefinariaState:
    print("\n💾 Salvando...")
    timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    
    # Chama a função utilitária de salvamento
    salvar_resultado(
        resultado=state["post"],
        tarefas=[], 
        entrada_usuario=state["input_text"],
        timestamp=timestamp
    )
    return {"timestamp": timestamp, "approved": True}

# --- 4. CONSTRUÇÃO DO GRAFO LANGGRAPH ---

graph = StateGraph(RefinariaState)

# Adicionando os nós
graph.add_node("analisar", analisar_estrutura)
graph.add_node("post", gerar_post)
graph.add_node("prompt_imagem", gerar_prompt_imagem)
graph.add_node("imagem", gerar_imagem)
graph.add_node("salvar", salvar)

# Ponto de entrada
graph.set_entry_point("analisar")

# Arestas lineares
graph.add_edge("analisar", "post")
graph.add_edge("post", "prompt_imagem")
graph.add_edge("prompt_imagem", "imagem")

# Aresta condicional (Humano decide após ver a imagem/post)
graph.add_conditional_edges(
    "imagem",          # Nó anterior
    revisar_input,     # Função que decide o próximo passo
    {
        "salvar": "salvar",
        "abortar": END
    }
)

graph.add_edge("salvar", END)

# Compilação
app = graph.compile()

# --- 5. EXECUÇÃO ---

if __name__ == "__main__":
    print("🚀 Iniciando Refinaria de Conteúdo LinkedIn")
    entrada_usuario = input("Cole o texto bruto ou ideia aqui: ")
    
    initial_state = {
        "input_text": entrada_usuario,
        "estrutura": None,
        "post": None,
        "image_prompt": None,
        "image_path": None,
        "approved": None,
        "timestamp": None
    }

    # Executa o grafo
    # O loop printa os estados conforme o grafo avança (opcional)
    for output in app.stream(initial_state):
        pass  # A lógica de print já está dentro das funções para manter o console limpo

    print("\n✅ Fluxo finalizado.")