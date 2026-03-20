import ollama

# Query with a task-specific instruction
task_description = "Identify clinical symptoms in medical text"
query = "Patient reports persistent cough and fatigue."

# Format with instruction for better performance
input_text = f"Instruct: {task_description}\nQuery: {query}"

response = ollama.embed(
    model='qwen3-embedding:4b',
    input=input_text
)

print(f"Vector Dimensions: {len(response.embeddings[0])}") # Default is 2560 for 4B


