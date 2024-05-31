PREFIX = """You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
You should use the tools below to answer the question posed of you:"""

FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""

SUFFIX = """Begin!

Question: {input}
Thought:{agent_scratchpad}"""

template  = "\n\n".join([PREFIX, "{tools}", FORMAT_INSTRUCTIONS, SUFFIX])


from langchain import hub

prompt = hub.pull("hwchase17/react")

print(prompt)