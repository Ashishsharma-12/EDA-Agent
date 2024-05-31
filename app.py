import os
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import login
from transformers import HfAgent
from gradio_tools import StableDiffusionPromptGeneratorTool
from transformers import Tool




load_dotenv(find_dotenv())

HF_TOKEN = os.environ.get('HUGGINGFACEHUB_API_TOKEN')

login(HF_TOKEN)


gradio_tool = StableDiffusionPromptGeneratorTool()

tool = Tool.from_gradio(gradio_tool)
agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder", additional_tools=[tool])

agent.run("Generate an image of the `prompt` after improving it.", prompt="A rabbit wearing a space suit")





















































