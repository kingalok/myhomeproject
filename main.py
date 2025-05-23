from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
import os

# Replace with your actual OpenAI API key.  Important: Keep this secure!
# You can set it as an environment variable:
# export OPENAI_API_KEY='your_api_key'
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

# 1. Define the LLM
llm = ChatOpenAI(model_name="gpt-4-turbo-preview", temperature=0)  # Using a powerful and capable model

# 2. Define the Tools
#   We'll start with a very simple tool: suggesting groceries.  Agentic behavior
#   will come from the LLM's ability to choose *when* to use this tool.
def suggest_groceries(preferences: str) -> str:
    """Suggests groceries based on user preferences.

    Args:
        preferences: A description of the user's dietary preferences and needs.

    Returns:
        A list of grocery items.
    """
    # This is a *very* basic implementation.  A real version would:
    # -  Have a much larger database of groceries.
    # -  Consider current offers, seasonality, etc.
    # -  Handle a wider range of preferences (allergies, etc.).
    if "vegan" in preferences.lower():
        return "Tofu, Almond milk, Broccoli, Quinoa, Spinach, Vegan cheese"
    elif "vegetarian" in preferences.lower():
        return "Eggs, Milk, Broccoli, Pasta, Spinach, Cheddar cheese"
    elif "low carb" in preferences.lower():
        return "Chicken breast, Salmon, Avocado, Broccoli, Olive oil, Eggs"
    elif "high protein" in preferences.lower():
        return "Chicken breast, Salmon, Protein powder, Greek yogurt, Lentils, Eggs"
    else:
        return "Bread, Milk, Eggs, Apples, Chicken, Rice"  # Default suggestion


tools = [
    Tool(
        name="grocery_suggestion",
        func=suggest_groceries,
        description="Suggests a list of groceries to buy based on your dietary preferences and needs.  Use this tool when the user asks for a grocery list, or asks what to buy.",
    )
]

# 3. Define the Prompt
#   The prompt is crucial for guiding the agent's behavior.  We use a structured
#   chat prompt to clearly define the agent's role, provide context, and
#   tell it how to use tools.
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            "You are a helpful shopping assistant. Your job is to understand the user's needs and suggest groceries. You have access to a tool called 'grocery_suggestion' which you can use to get a list of items to buy."
        ),
        MessagesPlaceholder(variable_name="chat_history"),  # Important for conversation history
        HumanMessage("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),  # Where the agent can record its thoughts
    ]
)

# 4. Create the Agent
#   We use a structured chat agent, which is designed to work well with
#   tools and structured prompts.
agent = create_structured_chat_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=None)  # No memory for simplicity

# 5. Run the Agent
#   This is where you interact with the agent.
def run_agent():
    """Runs the grocery shopping agent."""
    print("Welcome to the Grocery Shopping Assistant!")
    print("I can help you create a grocery list based on your preferences.")
    print("Type 'exit' to end the conversation.")

    chat_history = []  # Initialize chat history

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Construct messages, including chat history
        messages = [
            SystemMessage(
                "You are a helpful shopping assistant. Your job is to understand the user's needs and suggest groceries. You have access to a tool called 'grocery_suggestion' which you can use to get a list of items to buy."
            ),
        ]
        for h in chat_history:
            messages.append(h)
        messages.append(HumanMessage(content=user_input))
        try:
            result = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
            print("Assistant:", result["output"])
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(SystemMessage(content=result["output"]))

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_agent()
