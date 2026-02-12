import torch
from smolagents import CodeAgent, TransformersModel
from tools import GuestInfoRetrieverTool, WeatherInfoTool

model = TransformersModel(
    model_id="Qwen/Qwen2.5-3B-Instruct",
    trust_remote_code=True,
    max_new_tokens=512,
    temperature=0.2,
)

# Force fp16 on GPU (huge speedup)
model.model = model.model.half()
model.model.to("cuda")
model.model.eval()

# pad/eos config (removes warning)
tok = model.tokenizer
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model.model.config.pad_token_id = tok.pad_token_id
model.model.generation_config.pad_token_id = tok.pad_token_id
model.model.generation_config.eos_token_id = tok.eos_token_id


# Initialize the tools
guest_info_tool = GuestInfoRetrieverTool()
weather_tool = WeatherInfoTool()

# Create Alfred, our gala agent, with the guest info tool
alfred = CodeAgent(tools=[guest_info_tool, weather_tool], model = model)

# Part 1 Queries 
# print("----------------------------- Query 1 -----------------------------")
# response = alfred.run("Tell me about our guest named 'Lady Ada Lovelace'.")
# print("----------------------------- Query 1 -----------------------------")

# print("----------------------------- Query 2 -----------------------------")
# response = alfred.run("What do you know about 'Dr. Nikola Tesla'.")
# print("----------------------------- Query 2 -----------------------------")

# print("----------------------------- Query 3 -----------------------------")
# response = alfred.run("I'm about to talk to 'Marie Curie' what should I know.")
# print("----------------------------- Query 3 -----------------------------")

# Part 2 Queries
# print("----------------------------- Query 1 -----------------------------")
# query = "Tell me about 'Lady Ada Lovelace'"
# response = alfred.run(query)
# print("🎩 Alfred's Response:")
# print(response)
# print("----------------------------- Query 1 -----------------------------")
# print()

# print("----------------------------- Query 2 -----------------------------")
# query = "What's the weather like in Paris tonight? Will it be suitable for our fireworks display?"
# response = alfred.run(query)
# print("🎩 Alfred's Response:")
# print(response)
# print("----------------------------- Query 2 -----------------------------")
# print()

# print("----------------------------- Query 3 -----------------------------")
# query = "One of our guests is from Qwen. What can you tell me about their most popular model?"
# response = alfred.run(query)
# print("🎩 Alfred's Response:")
# print(response)
# print("----------------------------- Query 3 -----------------------------")
# print()

# print("----------------------------- Query 4 -----------------------------")
# query = "I need to speak with Dr. Nikola Tesla about recent advancements in wireless energy. Can you help me prepare for this conversation?"
# response = alfred.run(query)
# print("🎩 Alfred's Response:")
# print(response)
# print("----------------------------- Query 4 -----------------------------")
# print()

# print("----------------------------- Query 5 -----------------------------")
# query = "Tell me about our guest named 'Lady Ada Lovelace'."
# response = alfred.run(query)
# print("🎩 Alfred's Response:")
# print(response)
# print("----------------------------- Query 5 -----------------------------")
# print()

# print("----------------------------- Query 6 -----------------------------")
# query = "What do you know about 'Dr. Nikola Tesla'."
# response = alfred.run(query)
# print("🎩 Alfred's Response:")
# print(response)
# print("----------------------------- Query 6 -----------------------------")
# print()

# print("----------------------------- Query 7 -----------------------------")
# query = "I'm about to talk to 'Marie Curie' what should I know."
# response = alfred.run(query)
# print("🎩 Alfred's Response:")
# print(response)
# print("----------------------------- Query 7 -----------------------------")
# print()

# print("----------------------------- Query 8 -----------------------------")
# query = "I am going to be speaking with our guest Jonathan Rockwell what should I know?"
# response = alfred.run(query)
# print("🎩 Alfred's Response:")
# print(response)
# print("----------------------------- Query 8 -----------------------------")
# print()

print("----------------------------- Query 9 -----------------------------")
query = "Telle me everyting you know about our guest Sundar Majid."
response = alfred.run(query)
print("🎩 Alfred's Response:")
print(response)
print("----------------------------- Query 9 -----------------------------")
print()

print("----------------------------- Query 10 -----------------------------")
query = "Can you tell me everything you know about the guest Rarul Carnitez"
response = alfred.run(query)
print("🎩 Alfred's Response:")
print(response)
print("----------------------------- Query 10 -----------------------------")
print()



print("🎩 Alfred's Response:")
print(response)