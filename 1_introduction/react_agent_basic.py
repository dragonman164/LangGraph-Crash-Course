from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import initialize_agent, tool
from langchain_community.tools import TavilySearchResults
import datetime 

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
search_tool = TavilySearchResults(
    search_depth = "basic"
)

@tool
def get_system_time(format : str = "%Y-%m-%d %H:%M:%S"):
    """ Returns the current Date and Time in Specified format """

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time


# https://smith.langchain.com/hub/hwchase17/react
agent = initialize_agent(tools = [search_tool, 
                                  get_system_time], llm = llm, agent="zero-shot-react-description", 
                         verbose = True)

# agent.invoke("Give me a tweet about today's weather in Bangalore")

agent.invoke("When was the SpaceX's last launch. How many days have passed by ? ")

# result = llm.invoke("Give me a tweet about today's weather in Bangalore")
# print(result)