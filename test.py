from llm.response_generator import ResponseGenerator

generator = ResponseGenerator()

emotion = "sadness"
text = "I feel isolated and exhausted from work."

response = generator.generate(emotion, text)

print("Generated Response:\n")
print(response)