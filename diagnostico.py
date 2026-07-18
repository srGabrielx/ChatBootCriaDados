import os
import requests
from dotenv import load_dotenv, find_dotenv

def auditar_modelos_permitidos():
    """Interroga a API para listar os modelos ativos e disponíveis para a chave atual."""
    load_dotenv(find_dotenv())
    api_key = os.getenv("AIMLAPI_KEY")
    
    if not api_key:
        raise ValueError("Chave AIMLAPI_KEY não encontrada no ambiente.")

    # Endpoint de listagem de modelos (Padrão OpenAI)
    url = "https://api.aimlapi.com/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}

    print("Iniciando varredura no catálogo da AIMLAPI...")
    resposta = requests.get(url, headers=headers)

    if resposta.status_code != 200:
        print(f"Falha de autenticação ou roteamento: {resposta.status_code}")
        print(resposta.text)
        return

    dados = resposta.json()
    modelos = dados.get("data", [])
    
    if not modelos:
        print("A API retornou uma lista vazia. Sua conta pode não ter acesso a modelos ativos.")
        return

    print(f"\nSucesso. {len(modelos)} modelos detectados para a sua conta.\n")
    print("Modelos recomendados para Chat/RAG disponíveis:")
    
    # Filtra os resultados para exibir opções relevantes
    encontrou_chat = False
    for modelo in modelos:
        id_modelo = modelo.get("id", "")
        id_lower = id_modelo.lower()
        
        # Filtra por famílias de modelos eficientes para RAG
        if any(keyword in id_lower for keyword in ["llama", "mistral", "gemma", "qwen", "chat"]):
            print(f" -> {id_modelo}")
            encontrou_chat = True
            
    if not encontrou_chat:
        print("Nenhum modelo clássico de chat encontrado. Imprimindo os 10 primeiros disponíveis:")
        for modelo in modelos[:10]:
            print(f" -> {modelo.get('id')}")

if __name__ == "__main__":
    auditar_modelos_permitidos()