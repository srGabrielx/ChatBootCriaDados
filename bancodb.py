import os
import time
import shutil

from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Configurações Globais
PASTA_BASE = 'base'
PASTA_DB = "Filtro_db"

load_dotenv(find_dotenv())
AIMLAPI_KEY = os.getenv("AIMLAPI_KEY")

if not AIMLAPI_KEY:
    print("ALERTA: Variável AIMLAPI_KEY ausente do escopo de ambiente.")
else:
    print("DEBUG: Credencial AIMLAPI_KEY alocada em memória.")

# --- MÓDULO DE INGESTÃO E VETORIZAÇÃO ---

def carregar_documentos():
    """Extrai os artefatos PDF do diretório estipulado."""
    print(f"Inspecionando diretório: '{PASTA_BASE}'")
    carregador = PyPDFDirectoryLoader(PASTA_BASE, glob='*.pdf')
    try:
        documentos = carregador.load()
        print(f"Inventário concluído: {len(documentos)} artefato(s) detectado(s).")
        return documentos
    except Exception as anomalia:
        print(f"Falha na extração de dados: {anomalia}")
        return []

def dividir_chunks(documentos):
    """Fragmenta os artefatos documentais em porções textuais (chunks)."""
    print("Segmentando a base de conhecimento...")
    separador = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True
    )
    chunks = separador.split_documents(documentos)
    print(f"Fragmentação concluída: {len(chunks)} blocos gerados.")
    return chunks

def vetorizar_chunks(chunks):
    """Calcula os embeddings e persiste as matrizes no ChromaDB."""
    print(f"Iniciando projeção vetorial de {len(chunks)} matrizes...")
    
    try:
        inicio = time.time()
        funcao_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        if os.path.exists(PASTA_DB):
            print(f"Purgando persistência vetorial obsoleta em '{PASTA_DB}'...")
            shutil.rmtree(PASTA_DB)

        print(f"Consolidando novo repositório em '{PASTA_DB}'...")
        Chroma.from_documents(
            chunks,
            embedding=funcao_embedding,
            persist_directory=PASTA_DB
        )
        print(f"✅ Projeção e persistência finalizadas em {time.time() - inicio:.2f} s.\n"
             "Execute python main.py --chat "
              )
    except Exception as anomalia:
        print(f"Ruptura durante a vetorização: {anomalia}")
    
def criar_banco_de_dados():
    """Orquestra o pipeline de ETL (Extração, Transformação e Carga)."""
    print("Inicializando pipeline de indexação vetorial...")
    inicio = time.time()
    documentos = carregar_documentos()
    if not documentos:
        print("Execução abortada: Acervo documental incipiente.")
        return
    chunks = dividir_chunks(documentos)
    vetorizar_chunks(chunks)
    fim = time.time()
    print (f"Time {time.time() - inicio:.2f} s")

# --- MÓDULO DE INFERÊNCIA E RETRIEVAL ---

def configurar_infraestrutura_rag():
    """Instancia o banco vetorial, a topologia do LLM e a cadeia RAG LCEL."""
    inicio = time.time()
    
    if not os.path.exists(PASTA_DB):
        raise FileNotFoundError(f"Repositório '{PASTA_DB}' inacessível. \n"
                                f"Execute python main.py --create-db  \nTempo de falha: {time.time() - inicio:.2f} s")

    print("Carregando espaço latente (HuggingFace)...")
    funcao_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    banco_vetorial = Chroma(persist_directory=PASTA_DB, embedding_function=funcao_embedding)
    
    print("Instanciando cliente de inferência (AIMLAPI)...")
    cliente_llm = ChatOpenAI(
        api_key=AIMLAPI_KEY,
        base_url="https://api.aimlapi.com/v1",
      model="gpt-4.1-mini",
        temperature=0.3
    )

    # Transição para o paradigma LCEL (LangChain Expression Language)
    retriever = banco_vetorial.as_retriever(search_kwargs={'k': 5})
    
    # O paradigma atual exige um prompt estruturado para injetar o contexto recuperado
    prompt_contextual = ChatPromptTemplate.from_messages([
        ("assistant", "Atue como um analista de dados preciso. Responda à inquirição utilizando exclusivamente o contexto fornecido.\n\nContexto:\n{context}"),
        ("human", "{input}"),
    ])
    
    matriz_documentos = create_stuff_documents_chain(cliente_llm, prompt_contextual)
    matriz_rag = create_retrieval_chain(retriever, matriz_documentos)
    
    # Otimização: Remoção da variável 'fim' que não estava sendo utilizada
    print(f"Infraestrutura inicializada em {time.time() - inicio:.2f} s")
    return matriz_rag


def iniciar_chat():
    """Orquestra o laço interativo de transações conversacionais."""
    inicio = time.time()
    print("Inicializando interface terminal do RAG...")
    try:
        matriz_rag = configurar_infraestrutura_rag()
    except Exception as anomalia:
        print(f"Ruptura fatal na topologia da aplicação: {anomalia}")
        return

    print("\n🚀 Chat: Digite Sua Pergunta (ou 'sair').")
    while True:
        query = input("\n> ")
        if query.lower() in ('sair', 'exit', 'quit'):
            break
        if not query.strip():
            continue

        try:
            print("\n🧠 Computando inferência...")
           
            
            # LCEL: argumento primário é 'input', e o retorno reside em 'answer'
            resultado = matriz_rag.invoke({"input": query})
          
            
            print("\nResposta Sistêmica:")
            print(resultado.get("answer", "Inconsistência topológica: O modelo não propagou tensores de resposta."))
            print(f"\n(Tempo computacional: {time.time() - inicio:.2f} s)")

        except Exception as anomalia:
            print(f"Colapso durante a transação de rede: {anomalia}")
    fim = time.time()