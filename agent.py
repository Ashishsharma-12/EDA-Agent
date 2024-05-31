import os
from dotenv import load_dotenv, find_dotenv
from langchain.agents.react.agent import create_react_agent
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from langchain.agents import Tool
import pandas as pd
from langchain_core.prompts.prompt import PromptTemplate
from typing import Optional, Any, Dict, Union
from langchain.agents.agent import AgentExecutor, RunnableAgent, BaseMultiActionAgent, BaseSingleActionAgent
from langchain_experimental.tools.python.tool import PythonAstREPLTool
# from langchain.agents.format_scratchpad import format_log_to_str
from langchain_core.utils.interactive_env import is_interactive_env

# print("library loaded")

# HF login
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get('HUGGINGFACEHUB_API_TOKEN')

# print("login Successful")

# LLM
llm = HuggingFaceEndpoint(
        task="text-generation",
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        max_new_tokens=2048,
        top_k= 10,
        top_p= 0.9,
        temperature= 0.1,
        repetition_penalty=1.03,
        huggingfacehub_api_token= HF_TOKEN
    )

model = ChatHuggingFace(llm = llm)

# print('Model created')

# Helper Function to read files

def file_to_df(file_location: str):

    file_name = file_location.split("/")[-1]
    df = None
    if "csv" in file_name:
        df = pd.read_csv(file_location)
    elif "xlsx" in file_name:
        df = pd.read_excel(file_location)
    elif "json" in file_name:
        df = pd.read_json(file_location, orient="records")
    elif "parquet" in file_name:
        df = pd.read_parquet(file_location)
    elif "feather" in file_name:
        df = pd.read_feather(file_location)

    return df

# Agent function

def my_data_ananlysis_agent(model, filepath):

    # Read file

    df = file_to_df(filepath)

    # print("df created")

    # Define Tools

    search = DuckDuckGoSearchResults()

    search_tool = [
        Tool(
            name = "search",
            func=search.run,
            description="useful for when you need to answer questions using web search. You should ask targeted questions"
        )
    ]

    # Creating environment

    if is_interactive_env():
        pd.set_option("display.max_columns", None)

    locals = {"df": df}

    tools = search_tool + [PythonAstREPLTool(locals=locals)] 

    # Query template

    PREFIX = """
    You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
    You should use the tools below to answer the question posed of you:
    """

    FORMAT_INSTRUCTIONS = """
    Use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation/Validation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question, and print the generated tables in final answer\n
    """

    SUFFIX = """
    Begin!

    Question: {input}
    Thought:{agent_scratchpad}
    """

    template  = "".join([PREFIX, "\n{tools}\n", FORMAT_INSTRUCTIONS, SUFFIX])

    prompt = PromptTemplate.from_template(template)

    # Initialize agent

    agent: Union[BaseSingleActionAgent, BaseMultiActionAgent] = RunnableAgent(
                runnable=create_react_agent(llm = model, tools=tools, prompt=prompt), 
                input_keys_arg=["input"],
                return_keys_arg=["output"],
                stream_runnable = True
            )

    # agent = create_react_agent(llm = model, tools=tools, prompt=prompt)

    agent_executor_kwargs : Optional[Dict[str, Any]] = None

    data_agent = AgentExecutor(
            agent=agent,
            tools=tools,
            callback_manager=None,
            verbose=True,
            return_intermediate_steps=True,
            max_iterations=15,
            max_execution_time=None,
            early_stopping_method="force",
            handle_parsing_errors=True,
            **(agent_executor_kwargs or {}),
        )
    return data_agent

filepath = "data.csv"
my_agent = my_data_ananlysis_agent(model,filepath)
my_agent.invoke({"input":"correlation analysis?"})

























