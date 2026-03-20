import ollama

# Generate a standard 768-dimension embedding
response = ollama.embed(
    model='nomic-embed-text',
    input='Nomic-embed-text is an open-source long-context model.'
)

print(f"Embedding Dimension: {len(response.embeddings[0])}")

