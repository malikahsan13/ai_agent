from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

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
print(response)
