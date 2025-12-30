import os
from typing import TypedDict
from dotenv import load_dotenv

# LangGraph e LangChain imports
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# 1. Configuração Inicial
load_dotenv()

# Usaremos o Llama 3.1 70B para raciocínio (via Groq pela velocidade)
llm = ChatGroq(
    temperature=0.7, 
    model_name="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# 2. Definindo o Estado (A "Memória" da Empresa)
# Isso é o que passa de um agente para outro.
class AgentState(TypedDict):
    topic: str
    draft: str
    critique: str
    final_post: str

# 3. Definindo os Agentes (Os "Funcionários")

def writer_agent(state: AgentState):
    """Agente que cria o primeiro rascunho."""
    print(f"✍️  WRITER: Escrevendo sobre '{state['topic']}'...")
    
    prompt = f"Escreva um post curto e engajador para o LinkedIn sobre: {state['topic']}. Use um tom profissional mas provocativo."
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {"draft": response.content}

def critic_agent(state: AgentState):
    """Agente que critica o trabalho do writer."""
    print("🧐 CRITIC: Analisando o rascunho...")
    
    prompt = f"""
    Analise este rascunho de post: 
    '{state['draft']}'
    
    Liste 3 pontos críticos para melhorar a viralidade e clareza. Seja duro.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {"critique": response.content}

def refiner_agent(state: AgentState):
    """Agente que consolida a versão final."""
    print("🚀 REFINER: Polindo a versão final...")
    
    prompt = f"""
    Você é um editor expert.
    Rascunho Original: {state['draft']}
    Críticas recebidas: {state['critique']}
    
    Reescreva o post final incorporando as melhorias. O resultado deve ser pronto para publicar.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {"final_post": response.content}

# 4. Construindo o Grafo (O "Workflow")

workflow = StateGraph(AgentState)

# Adicionando os nós (Nodes)
workflow.add_node("writer", writer_agent)
workflow.add_node("critic", critic_agent)
workflow.add_node("refiner", refiner_agent)

# Definindo o fluxo (Edges)
# Start -> Writer -> Critic -> Refiner -> End
workflow.set_entry_point("writer")
workflow.add_edge("writer", "critic")
workflow.add_edge("critic", "refiner")
workflow.add_edge("refiner", END)

# Compilando a "Empresa"
app = workflow.compile()

# 5. Execução (O "Trigger")

if __name__ == "__main__":
    topic = input("Sobre o que você quer postar hoje? (ex: AI Agents): ")
    
    inputs = {"topic": topic}
    
    # Rodando o grafo
    result = app.invoke(inputs)
    
    print("\n" + "="*50)
    print("RESULTADO FINAL (PRONTO PARA O MILHÃO):")
    print("="*50 + "\n")
    print(result['final_post'])