### Install llama-cpp-python and huggingface-hub prior to execution. ###

from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
	filename="Llama-3.2-1B-Instruct-f16.gguf",
)

completion = llm.create_chat_completion(
	messages = [
		{
			"role": "user",
			"content": "What is the capital of France?"
		}
	]
)

print("Chat completion: %s", completion)
