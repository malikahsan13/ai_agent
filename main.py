from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

# llm = ChatOpenAI(model="gpt-40-mini")

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm2 = ChatAnthropic(model="claude-3-5-sonnet-20241022")

response = llm2.invoice("What is the meaning of life?")

parser = PydantincOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tolls.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """
        ),
        ("placeholder","{chat_history}"),
        ("human","{query}"),
        ("placeholder","{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())
#print(response)

tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("What can i help you research ? ")
raw_response = agent_executor.invoke({"query":query})

try:
    structured_reponse = parser.parse(raw_response.get("output")[0]["text"])
except Exception as e:
    print("error parsing resposne",e,"Raw Response - ", raw_response)
