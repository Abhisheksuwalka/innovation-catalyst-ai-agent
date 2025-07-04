import os
from smolagents import CodeAgent, DuckDuckGoSearchTool
from smolagents.models import LiteLLMModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set your GitHub token
# os.environ["GITHUB_API_KEY"] = "your_code_here"  # Replace with your actual token

# Initialize the model using the correct GitHub Models format
model = LiteLLMModel(
    model_id="github/microsoft/phi-4-multimodal-instruct",  # Correct format: github/{publisher}/{model_name}
    api_key=os.environ["GITHUB_API_KEY"],
    api_base="https://models.github.ai/inference",
    temperature=0.7,
    max_tokens=1000
)

# Create tools (optional)
search_tool = DuckDuckGoSearchTool()

# Initialize the CodeAgent
agent = CodeAgent(
    tools=[search_tool],
    model=model,
    max_steps=10
)

# Test the agent
try:
    result = agent.run("What is the capital of France?")
    print("✅ Success!")
    print(result)
except Exception as e:
    print(f"❌ Error: {e}")
