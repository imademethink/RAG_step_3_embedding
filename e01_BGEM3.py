import ollama

# Generate embeddings for a single string or list of strings
response = ollama.embed(
    model='bge-m3',
    input='BGE-M3 supports dense, sparse, and multi-vector retrieval.'
    # input='"Patient presents with a persistent, non-productive cough and progressive dyspnea over three weeks. '
    #       'Clinical examination reveals bilateral basal crackles and decreased oxygen saturation at 92% on room air. '
    #       'Relevant medical history includes controlled hypertension and a decade of secondary tobacco exposure. '
    #       'Differential diagnoses consider community-acquired pneumonia, '
    #       'pulmonary edema, or interstitial lung disease. Preliminary laboratory results indicate elevated '
    #       'C-reactive protein levels and mild leukocytosis, while a chest X-ray shows diffuse reticular opacities. '
    #       'Following standardized protocols, a high-resolution CT scan is scheduled to evaluate parenchymal '
    #       'involvement. Immediate management includes supplemental oxygen therapy and '
    #       'empiric antibiotic administration pending sputum culture results."'
)

# Output is a list of 1024-dimensional vectors
print(f"Embedding Dimension: {len(response.embeddings[0])}")
print(f"Sample values: {response.embeddings[0][:5]}")
