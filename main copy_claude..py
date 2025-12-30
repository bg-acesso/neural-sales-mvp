"""
Refinaria de Conteúdo - Sistema de Criação Automatizada para LinkedIn
Versão: 2.0 (Melhorada)
"""

import os
import sys
import time
import json
import shutil
import requests
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from tenacity import retry, stop_after_attempt, wait_exponential

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

class Config:
    """Configurações centralizadas do sistema"""
    
    # Carrega variáveis de ambiente
    load_dotenv()
    
    HF_TOKEN = os.getenv("HF_TOKEN")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # Modelos
    LLM_MODEL = "groq/llama-3.3-70b-versatile"
    IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"
    
    # Diretórios
    OUTPUT_DIR = Path("galeria_refinaria")
    LOGS_DIR = Path("logs")
    
    # Limites
    MAX_INPUT_LENGTH = 10000
    MIN_INPUT_LENGTH = 50
    IMAGE_TIMEOUT = 60
    MAX_RETRIES = 3
    MAX_RPM = 10
    
    # API URLs
    HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{IMAGE_MODEL}"
    
    @classmethod
    def validate(cls):
        """Valida configurações essenciais"""
        if not cls.HF_TOKEN or not cls.GROQ_API_KEY:
            raise EnvironmentError(
                "\n❌ ERRO: Configuração incompleta!\n\n"
                "Configure as seguintes variáveis no arquivo .env:\n"
                "  HF_TOKEN=hf_...\n"
                "  GROQ_API_KEY=gsk_...\n\n"
                "Obtenha suas chaves em:\n"
                "  - HuggingFace: https://huggingface.co/settings/tokens\n"
                "  - Groq: https://console.groq.com/keys"
            )
        
        # Cria diretórios necessários
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
        cls.LOGS_DIR.mkdir(exist_ok=True)

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging():
    """Configura sistema de logging"""
    log_file = Config.LOGS_DIR / f"refinaria_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================================
# FERRAMENTAS
# ============================================================================

class FluxImageTool(BaseTool):
    """Gerador de imagens usando Flux via HuggingFace"""
    
    name: str = "Gerador de Imagem Flux"
    description: str = "Gera imagens realistas e artísticas baseadas em prompts descritivos."
    
    @retry(
        stop=stop_after_attempt(Config.MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _run(self, prompt: str) -> str:
        """Gera imagem com retry automático"""
        prompt_limpo = str(prompt).strip().replace('"', '').replace("'", "")
        
        logger.info(f"🎨 Gerando imagem: {prompt_limpo[:60]}...")
        
        headers = {"Authorization": f"Bearer {Config.HF_TOKEN}"}
        
        try:
            response = requests.post(
                Config.HF_API_URL,
                headers=headers,
                json={"inputs": prompt_limpo},
                timeout=Config.IMAGE_TIMEOUT
            )
            
            if response.status_code == 200:
                timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
                nome_arquivo = Config.OUTPUT_DIR / f"post_{timestamp}.png"
                
                with open(nome_arquivo, "wb") as f:
                    f.write(response.content)
                
                logger.info(f"✅ Imagem salva: {nome_arquivo}")
                return str(nome_arquivo.absolute())
            
            else:
                error_msg = f"Erro {response.status_code}: {response.text[:200]}"
                logger.error(f"❌ Falha na API de imagem: {error_msg}")
                return f"Erro ao gerar imagem: {error_msg}"
                
        except requests.exceptions.Timeout:
            logger.warning("⏱️ Timeout na geração de imagem. Tentando novamente...")
            raise
        except Exception as e:
            logger.error(f"❌ Erro inesperado: {str(e)}")
            return f"Erro ao gerar imagem: {str(e)}"

# ============================================================================
# CAPTURA DE INPUT
# ============================================================================

def capturar_materia_prima() -> str:
    """Captura input do usuário com validação"""
    print("\n" + "="*70)
    print("📝 TERMINAL DE CRIAÇÃO DE CONTEÚDO".center(70))
    print("="*70)
    print("\nDigite ou cole sua ideia, rascunho ou transcrição abaixo.")
    print("💡 Dica: Quanto mais contexto você fornecer, melhor o resultado.")
    print("\n⚠️  Para finalizar: digite 'FIM' em uma nova linha e pressione Enter.\n")
    print("-" * 70)
    
    linhas = []
    try:
        while True:
            linha = input()
            if linha.strip().upper() == 'FIM':
                break
            linhas.append(linha)
    except (EOFError, KeyboardInterrupt):
        print("\n\n⚠️  Captura interrompida.")
        sys.exit(0)
    
    texto_final = "\n".join(linhas).strip()
    
    # Validações
    if not texto_final:
        logger.error("❌ Nenhum texto inserido.")
        print("\n❌ Você precisa inserir algum conteúdo. Tente novamente.\n")
        return capturar_materia_prima()
    
    if len(texto_final) < Config.MIN_INPUT_LENGTH:
        logger.warning(f"Texto muito curto: {len(texto_final)} caracteres")
        print(f"\n⚠️  Texto muito curto ({len(texto_final)} caracteres).")
        print(f"    Mínimo recomendado: {Config.MIN_INPUT_LENGTH} caracteres.\n")
        
        continuar = input("Deseja continuar mesmo assim? (s/n): ").lower()
        if continuar != 's':
            return capturar_materia_prima()
    
    if len(texto_final) > Config.MAX_INPUT_LENGTH:
        logger.warning(f"Texto truncado de {len(texto_final)} para {Config.MAX_INPUT_LENGTH}")
        print(f"\n⚠️  Texto muito longo. Será truncado para {Config.MAX_INPUT_LENGTH} caracteres.")
        texto_final = texto_final[:Config.MAX_INPUT_LENGTH]
    
    logger.info(f"✅ Input capturado: {len(texto_final)} caracteres")
    return texto_final

def carregar_de_arquivo(caminho: str) -> str:
    """Carrega conteúdo de um arquivo"""
    try:
        with open(caminho, 'r', encoding='utf-8') as f:
            conteudo = f.read().strip()
        
        if len(conteudo) > Config.MAX_INPUT_LENGTH:
            logger.warning(f"Arquivo truncado de {len(conteudo)} para {Config.MAX_INPUT_LENGTH}")
            conteudo = conteudo[:Config.MAX_INPUT_LENGTH]
        
        logger.info(f"✅ Arquivo carregado: {caminho} ({len(conteudo)} caracteres)")
        return conteudo
    
    except FileNotFoundError:
        logger.error(f"❌ Arquivo não encontrado: {caminho}")
        print(f"\n❌ Arquivo não encontrado: {caminho}\n")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Erro ao ler arquivo: {str(e)}")
        print(f"\n❌ Erro ao ler arquivo: {str(e)}\n")
        sys.exit(1)

# ============================================================================
# AGENTES
# ============================================================================

def criar_agentes() -> Dict[str, Agent]:
    """Cria e retorna todos os agentes do sistema"""
    
    agentes = {
        'analista': Agent(
            role='Estrategista de Conteúdo',
            goal='Transformar informações caóticas em estrutura narrativa clara e persuasiva.',
            backstory=(
                'Você é um estrategista experiente que analisa conteúdo bruto e identifica '
                'os elementos-chave: gancho emocional, problema central, solução proposta e '
                'call-to-action. Você organiza ideias em frameworks que maximizam engajamento.'
            ),
            verbose=True,
            llm=Config.LLM_MODEL
        ),
        
        'copywriter': Agent(
            role='Copywriter LinkedIn Expert',
            goal='Escrever posts virais que geram engajamento e autoridade profissional.',
            backstory=(
                'Você é um copywriter especializado em LinkedIn com anos de experiência. '
                'Seu estilo: parágrafos curtos, frases impactantes, tom executivo mas acessível. '
                'Você sabe usar espaçamento, quebras de linha e estrutura para manter atenção. '
                'Evita clichês corporativos e foca em storytelling autêntico.'
            ),
            verbose=True,
            llm=Config.LLM_MODEL
        ),
        
        'designer': Agent(
            role='Diretor de Arte Digital',
            goal='Criar conceitos visuais que traduzem ideias complexas em imagens memoráveis.',
            backstory=(
                'Você é um diretor de arte com expertise em design digital e tech aesthetics. '
                'Você pensa em termos de composição, paleta de cores, estilo visual e mood. '
                'Seus prompts são concisos, específicos e usam vocabulário de design profissional: '
                '"isometric 3d render", "minimalist illustration", "gradient background", '
                '"corporate tech aesthetic". Você evita clichês visuais e busca originalidade.'
            ),
            verbose=True,
            llm=Config.LLM_MODEL
        ),
        
        'executor': Agent(
            role='Renderizador de Imagens',
            goal='Executar a geração técnica de imagens com precisão.',
            backstory='Você é o executor técnico que transforma conceitos visuais em arquivos de imagem.',
            verbose=True,
            llm=Config.LLM_MODEL,
            tools=[FluxImageTool()]
        )
    }
    
    logger.info(f"✅ {len(agentes)} agentes criados")
    return agentes

# ============================================================================
# TAREFAS
# ============================================================================

def criar_tarefas(entrada_usuario: str, agentes: Dict[str, Agent]) -> list:
    """Cria a pipeline de tarefas"""
    
    t1_estrutura = Task(
        description=(
            f'Analise este conteúdo bruto e defina a melhor estrutura para um post de LinkedIn:\n\n'
            f'"""{entrada_usuario}"""\n\n'
            f'Identifique:\n'
            f'1. GANCHO: Como capturar atenção nas primeiras linhas?\n'
            f'2. PROBLEMA/CONTEXTO: Qual dor ou situação está sendo abordada?\n'
            f'3. SOLUÇÃO/INSIGHT: Qual é a mensagem central ou lição?\n'
            f'4. CALL-TO-ACTION: Como engajar o leitor ao final?\n'
            f'5. TOM: Qual abordagem emocional funciona melhor? (inspirador, técnico, storytelling, etc.)'
        ),
        expected_output='Estrutura detalhada em tópicos com recomendações estratégicas.',
        agent=agentes['analista']
    )
    
    t2_post = Task(
        description=(
            'Escreva o post completo para LinkedIn seguindo a estrutura definida. '
            'REGRAS ESSENCIAIS:\n'
            '- Máximo 1.300 caracteres (LinkedIn tem limite de 3.000, mas ideal é mais conciso)\n'
            '- Primeira linha DEVE prender atenção (use gancho forte)\n'
            '- Parágrafos de 1-2 linhas no máximo\n'
            '- Use espaçamento estratégico para respiração visual\n'
            '- Inclua 3-5 hashtags relevantes ao final (separadas por espaços)\n'
            '- Tom profissional mas conversacional\n'
            '- Evite: emojis excessivos, clichês corporativos, jargão desnecessário\n'
            '- Finalize com pergunta ou call-to-action que incentive comentários'
        ),
        expected_output='Post completo formatado e pronto para publicação no LinkedIn.',
        agent=agentes['copywriter'],
        context=[t1_estrutura]
    )
    
    t3_prompt = Task(
        description=(
            'Crie um prompt de imagem em INGLÊS (máximo 15 palavras) que capture visualmente '
            'a essência do post. Use vocabulário de design profissional.\n\n'
            'ESTILOS SUGERIDOS:\n'
            '- "isometric 3d render"\n'
            '- "minimalist tech illustration"\n'
            '- "abstract corporate design"\n'
            '- "gradient modern aesthetic"\n'
            '- "geometric shapes"\n\n'
            'EVITE: texto na imagem, rostos, logos, elementos muito literais.\n'
            'FOQUE: em conceitos abstratos, composição visual forte, paleta moderna.'
        ),
        expected_output='Prompt de imagem em inglês, conciso e visualmente descritivo (máx 15 palavras).',
        agent=agentes['designer'],
        context=[t2_post]
    )
    
    t4_render = Task(
        description='Gere a imagem usando o prompt visual criado.',
        expected_output='Caminho absoluto do arquivo de imagem gerado.',
        agent=agentes['executor'],
        context=[t3_prompt]
    )
    
    logger.info("✅ Pipeline de 4 tarefas configurada")
    return [t1_estrutura, t2_post, t3_prompt, t4_render]

# ============================================================================
# SALVAMENTO E METADATA
# ============================================================================

def salvar_resultado(resultado: Any, tarefas: list, entrada_usuario: str, timestamp: str) -> Dict[str, Path]:
    """Salva o resultado completo com metadata em subpasta dedicada"""
    
    # Cria subpasta para este post
    pasta_post = Config.OUTPUT_DIR / f"post_{timestamp}"
    pasta_post.mkdir(exist_ok=True)
    logger.info(f"📁 Criando pasta: {pasta_post}")
    
    arquivos = {}
    
    # Extrai o conteúdo do post
    post_final = str(tarefas[1].output.raw) if hasattr(tarefas[1].output, 'raw') else str(tarefas[1].output)
    
    # Salva o post em texto
    arquivo_post = pasta_post / "post.txt"
    with open(arquivo_post, 'w', encoding='utf-8') as f:
        f.write(post_final)
    arquivos['post'] = arquivo_post
    logger.info(f"✅ Post salvo: {arquivo_post}")
    
    # Salva o input original
    arquivo_input = pasta_post / "input_original.txt"
    with open(arquivo_input, 'w', encoding='utf-8') as f:
        f.write(entrada_usuario)
    arquivos['input'] = arquivo_input
    logger.info(f"✅ Input original salvo: {arquivo_input}")
    
    # Salva a estrutura/análise
    estrutura = str(tarefas[0].output.raw) if hasattr(tarefas[0].output, 'raw') else str(tarefas[0].output)
    arquivo_estrutura = pasta_post / "estrutura.txt"
    with open(arquivo_estrutura, 'w', encoding='utf-8') as f:
        f.write(estrutura)
    arquivos['estrutura'] = arquivo_estrutura
    logger.info(f"✅ Estrutura salva: {arquivo_estrutura}")
    
    # Salva o prompt da imagem
    prompt_img = str(tarefas[2].output.raw) if hasattr(tarefas[2].output, 'raw') else str(tarefas[2].output)
    arquivo_prompt = pasta_post / "prompt_imagem.txt"
    with open(arquivo_prompt, 'w', encoding='utf-8') as f:
        f.write(prompt_img)
    arquivos['prompt'] = arquivo_prompt
    logger.info(f"✅ Prompt de imagem salvo: {arquivo_prompt}")
    
    # Move a imagem para a subpasta (se existir)
    caminho_imagem_original = str(tarefas[3].output.raw) if hasattr(tarefas[3].output, 'raw') else str(tarefas[3].output)
    if os.path.exists(caminho_imagem_original):
        import shutil
        arquivo_imagem = pasta_post / "imagem.png"
        shutil.move(caminho_imagem_original, arquivo_imagem)
        arquivos['imagem'] = arquivo_imagem
        logger.info(f"✅ Imagem movida para: {arquivo_imagem}")
    
    # Salva metadata completa
    metadata = {
        "timestamp": timestamp,
        "data_criacao": datetime.now().isoformat(),
        "input": {
            "tamanho_caracteres": len(entrada_usuario),
            "arquivo": "input_original.txt"
        },
        "post_final": {
            "tamanho_caracteres": len(post_final),
            "arquivo": "post.txt"
        },
        "estrutura": {
            "arquivo": "estrutura.txt"
        },
        "imagem": {
            "prompt": prompt_img,
            "arquivo": "imagem.png",
            "prompt_arquivo": "prompt_imagem.txt"
        },
        "config": {
            "modelo_llm": Config.LLM_MODEL,
            "modelo_imagem": Config.IMAGE_MODEL
        },
        "arquivos": {
            "post": "post.txt",
            "input": "input_original.txt",
            "estrutura": "estrutura.txt",
            "imagem": "imagem.png",
            "prompt": "prompt_imagem.txt",
            "metadata": "metadata.json"
        }
    }
    
    arquivo_meta = pasta_post / "metadata.json"
    with open(arquivo_meta, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    arquivos['metadata'] = arquivo_meta
    logger.info(f"✅ Metadata salva: {arquivo_meta}")
    
    # Cria um README na pasta
    arquivo_readme = pasta_post / "README.md"
    with open(arquivo_readme, 'w', encoding='utf-8') as f:
        f.write(f"""# Post criado em {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}

## 📄 Arquivos

- **post.txt**: Post final pronto para LinkedIn
- **imagem.png**: Imagem de capa gerada
- **input_original.txt**: Seu rascunho/ideia original
- **estrutura.txt**: Análise estratégica do conteúdo
- **prompt_imagem.txt**: Prompt usado para gerar a imagem
- **metadata.json**: Informações técnicas completas

## 📊 Estatísticas

- **Input**: {len(entrada_usuario)} caracteres
- **Post final**: {len(post_final)} caracteres
- **Modelo LLM**: {Config.LLM_MODEL}
- **Modelo Imagem**: {Config.IMAGE_MODEL}

## 🚀 Como usar

1. Copie o conteúdo de `post.txt`
2. Cole no LinkedIn
3. Anexe a imagem `imagem.png`
4. Publique!
""")
    arquivos['readme'] = arquivo_readme
    logger.info(f"✅ README criado: {arquivo_readme}")
    
    return arquivos

# ============================================================================
# PREVIEW E CONFIRMAÇÃO
# ============================================================================

def mostrar_preview(tarefas: list) -> bool:
    """Mostra preview e pede confirmação"""
    
    post_final = str(tarefas[1].output.raw) if hasattr(tarefas[1].output, 'raw') else str(tarefas[1].output)
    
    print("\n" + "="*70)
    print("📄 PREVIEW DO POST".center(70))
    print("="*70 + "\n")
    
    print(post_final[:800] + "..." if len(post_final) > 800 else post_final)
    
    print("\n" + "-"*70)
    print(f"📊 Tamanho: {len(post_final)} caracteres")
    print("-"*70 + "\n")
    
    try:
        resposta = input("💾 Salvar este conteúdo? (s/n): ").lower().strip()
        return resposta == 's'
    except (EOFError, KeyboardInterrupt):
        return False

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Função principal"""
    
    # Parse argumentos
    parser = argparse.ArgumentParser(
        description="Refinaria de Conteúdo - Sistema de Criação para LinkedIn"
    )
    parser.add_argument(
        "--file", "-f",
        help="Processar conteúdo de um arquivo em vez de input interativo"
    )
    parser.add_argument(
        "--skip-preview", "-y",
        action="store_true",
        help="Pular preview e salvar automaticamente"
    )
    args = parser.parse_args()
    
    # Banner
    print("\n" + "="*70)
    print("🏭 REFINARIA DE CONTEÚDO v2.0".center(70))
    print("="*70)
    
    try:
        # Valida configuração
        Config.validate()
        logger.info("✅ Configuração validada")
        
        # Captura input
        if args.file:
            entrada_usuario = carregar_de_arquivo(args.file)
        else:
            entrada_usuario = capturar_materia_prima()
        
        print(f"\n⚙️  PROCESSANDO {len(entrada_usuario)} CARACTERES...")
        logger.info(f"Iniciando processamento de {len(entrada_usuario)} caracteres")
        
        # Cria agentes e tarefas
        agentes = criar_agentes()
        tarefas = criar_tarefas(entrada_usuario, agentes)
        
        # Executa a crew
        print("\n🚀 INICIANDO PIPELINE DE CRIAÇÃO...\n")
        crew = Crew(
            agents=list(agentes.values()),
            tasks=tarefas,
            verbose=True,
            max_rpm=Config.MAX_RPM
        )
        
        resultado = crew.kickoff()
        
        # Preview e confirmação
        if not args.skip_preview:
            if not mostrar_preview(tarefas):
                print("\n❌ Operação cancelada pelo usuário.\n")
                logger.info("Operação cancelada pelo usuário")
                return
        
        # Salva resultado
        timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        arquivos = salvar_resultado(resultado, tarefas, entrada_usuario, timestamp)
        
        # Sucesso
        pasta_post = Config.OUTPUT_DIR / f"post_{timestamp}"
        print("\n" + "="*70)
        print("✅ CONTEÚDO PRONTO PARA PUBLICAR".center(70))
        print("="*70)
        print(f"\n📁 Pasta do post: {pasta_post.absolute()}")
        print(f"\n📄 Arquivos criados:")
        print(f"   • post.txt           - Post pronto para LinkedIn")
        print(f"   • imagem.png         - Imagem de capa")
        print(f"   • input_original.txt - Seu rascunho original")
        print(f"   • estrutura.txt      - Análise estratégica")
        print(f"   • prompt_imagem.txt  - Prompt da imagem")
        print(f"   • metadata.json      - Dados técnicos")
        print(f"   • README.md          - Guia de uso")
        print()
        
        logger.info("✅ Processamento concluído com sucesso")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Operação interrompida pelo usuário.\n")
        logger.warning("Operação interrompida pelo usuário")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n❌ ERRO: {str(e)}\n")
        logger.error(f"Erro fatal: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()