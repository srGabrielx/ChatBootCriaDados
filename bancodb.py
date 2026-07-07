import os 
import time
import shutil

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, find_dotenv 
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain.chains import RetrievalQA

PASTA_BASE = 'base'
PASTA_DB = "db_google"

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv(find_dotenv())
AIMLAPI_KEY = os.getenv("AIMLAPI_KEY")

# Mensagem de depuração para verificar se a chave foi carregada
if not AIMLAPI_KEY:
    print("AVISO: A variável de ambiente AIMLAPI_KEY não foi encontrada.")
    print("Certifique-se de que você tem um arquivo .env no mesmo diretório com o conteúdo: AIMLAPI_KEY='sua_chave_aqui'")
else:
    print("DEBUG: Chave AIMLAPI_KEY carregada com sucesso.")


def criar_banco_de_dados():
   """
   Carrega os documentos PDF, divide-os em chunks e os vetoriza,
   salvando em um banco de dados vetorial Chroma.
   """
   print("Iniciando a criação do banco de dados vetorial...")
   documentos = carregar_documentos()
   if not documentos:
       print("Nenhum documento PDF encontrado na pasta 'base'. O banco de dados não será criado.")
       return
   chunks = dividir_chunks(documentos)
   vetorizar_chunks(chunks)
   
def carregar_documentos():
    """Carrega os arquivos PDF da pasta especificada."""
    print(f"Carregando documentos da pasta: '{PASTA_BASE}'")
    carregador = PyPDFDirectoryLoader(PASTA_BASE, glob='*.pdf')
    try:
        documentos = carregador.load()
        print(f"Encontrados {len(documentos)} documento(s).")
        return documentos
    except Exception as e:
        print(f"ERRO ao carregar documentos: {e}")
        return []


def dividir_chunks(documentos):
 """Divide os documentos em chunks menores."""
 print("Dividindo documentos em chunks...")
 separador_documentos = RecursiveCharacterTextSplitter( 
    chunk_size = 1000,
    chunk_overlap = 200, # Reduzi o overlap para um valor mais comum
    length_function = len,
    add_start_index = True
 )
 chunks = separador_documentos.split_documents(documentos)
 print(f"Documentos divididos em {len(chunks)} chunks.")
 return chunks

def vetorizar_chunks(chunks):
    """Vetoriza os chunks usando um modelo local e os salva no ChromaDB."""
    print(f"Iniciando a vetorização de {len(chunks)} chunks usando Sentence Transformers...")
    start_time = time.time()

    try:
        # Modelo de embedding que será executado localmente
        embedding_hf = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Remove o diretório do banco de dados antigo se ele existir
        if os.path.exists(PASTA_DB):
            print(f"Removendo banco de dados antigo em '{PASTA_DB}'...")
            shutil.rmtree(PASTA_DB)

        # Cria e persiste o novo banco de dados
        print(f"Criando e persistindo o banco de dados em '{PASTA_DB}'...")
        db = Chroma.from_documents(
            chunks,
            embedding=embedding_hf,
            persist_directory=PASTA_DB
        )

        end_time = time.time()
        print(f"✅ Vetorização e criação do DB concluídas em {end_time - start_time:.2f} segundos.")

    except Exception as e:
        print(f"ERRO durante a vetorização ou criação do DB: {e}")


def iniciar_chat():
    """Inicia o chat interativo com o modelo de linguagem."""
    print("\nIniciando o sistema de chat...")

    # 1. Verifica se o banco de dados vetorial existe
    if not os.path.exists(PASTA_DB):
        print(f"ERRO: Banco de dados '{PASTA_DB}' não encontrado.")
        print("Por favor, execute o script com o argumento '--create-db' primeiro.")
        return

    # 2. Carrega o banco de dados vetorial e a função de embedding
    print("Carregando base de vetores (usando Sentence Transformers)...")
    try:
        embedding_hf = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = Chroma(persist_directory=PASTA_DB, embedding_function=embedding_hf)
        print("Base de dados carregada com sucesso!")
    except Exception as e:
        print(f"ERRO ao carregar banco de dados ou embeddings: {e}")
        return

    # 3. Configura a conexão com a API (AIMLAPI)
    try:
        print("Configurando a conexão com a AIMLAPI...")
        aimlapi_base_url = "https://api.aimlapi.com"
        aimlapi_model = "mistralai/Mistral-7B-Instruct-v0.2"

        
        # As versões mais recentes do LangChain usam 'api_key', 'base_url' e 'model'
        aimlapi_model = "mistralai/Mistral-7B-Instruct-v0.2" # Verifique o nome do modelo

        llm = ChatOpenAI(
            api_key=AIMLAPI_KEY,
            base_url=aimlapi_base_url,
            model=aimlapi_model,
            temperature=0.3
        )
        print(f" Conexão com AIMLAPI (modelo '{aimlapi_model}') estabelecida!")

    except Exception as e:
        print(f"ERRO FATAL ao configurar a conexão com a API: {e}")
        return

    # 4. Cria a cadeia de Pergunta e Resposta (QA Chain)
    retriever = db.as_retriever(search_kwargs={'k': 5})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True # Opcional: para ver os documentos fonte
    )

    # 5. Inicia o loop de chat
    print("\n🚀 Sistema pronto! Faça sua pergunta ou digite 'sair' para terminar.")
    while True:
        query = input("\n> ")
        if query.lower() in ('sair', 'exit', 'quit'):
            break
        if not query.strip():
            continue

        try:
            print("\n🧠 Pensando...")
            start = time.time()
            result = qa_chain.invoke({"query": query})
            end = time.time()

            print("\nResposta:")
            print(result.get("result", "Nenhuma resposta recebida da API."))
            print(f"\n(Tempo de resposta: {end - start:.2f} segundos)")

        except Exception as e:
            print(f"Ocorreu um erro durante a consulta à API: {e}")
