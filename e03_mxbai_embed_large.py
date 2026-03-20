import ollama

query = "Represent this sentence for searching relevant passages: What is the weather?"
response = ollama.embed(model='mxbai-embed-large', input=query)
print(len(response.embeddings[0])) # 1024
