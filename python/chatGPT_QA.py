import os
import openai

if "OPENAI_API_KEY" not in os.environ:
    default_key_file = os.path.expanduser("~/.openai_key")
    if not os.path.exists(default_key_file):
        raise ValueError("Please set the OPENAI_API_KEY environment variable or create a file at "
                         "~/.openai_key with your OpenAI API key.")
    with open(default_key_file) as f:
        openai.api_key = f.read().strip()
else:
    openai.api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is human life expectancy in the United States?"

response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    temperature=0,
    max_tokens=100,
    top_p=1,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=["\n"]
)
