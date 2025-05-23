import os
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage # Import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain_openai import ChatOpenAI

# --- 1. Configuration and LLM Initialization ---

openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError(
        "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
    )

llm = ChatOpenAI(model_name="gpt-4-turbo-preview", temperature=0)

# --- 2. Define the Tools ---

def suggest_groceries(preferences: str) -> str:
    """Suggests a list of groceries based on user preferences.

    Args:
        preferences: A description of the user's dietary preferences and needs.

    Returns:
        A string containing a comma-separated list of suggested grocery items.
    """
    preferences_lower = preferences.lower()
    if "vegan" in preferences_lower:
        return "Tofu, Almond milk, Broccoli, Quinoa, Spinach, Vegan cheese"
    elif "vegetarian" in preferences_lower:
        return "Eggs, Milk, Broccoli, Pasta, Spinach, Cheddar cheese"
    elif "low carb" in preferences_lower:
        return "Chicken breast, Salmon, Avocado, Broccoli, Olive oil, Eggs"
    elif "high protein" in preferences_lower:
        return "Chicken breast, Salmon, Protein powder, Greek yogurt, Lentils, Eggs"
    else:
        return "Bread, Milk, Eggs, Apples, Chicken, Rice"

tools = [
    Tool(
        name="grocery_suggestion",
        func=suggest_groceries,
        description="""Useful for suggesting a list of groceries to buy.
        Use this tool when the user asks for a grocery list, or asks what to buy.
        The input to this tool should be a string describing the user's dietary preferences or needs.""",
    )
]

# --- 3. Define the Prompt Template ---

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            "You are a helpful shopping assistant. Your job is to understand the user's needs and suggest groceries using the tools available to you."
        ),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        HumanMessage("{input}"),
        # This placeholder is for the agent's internal thoughts and tool interactions.
        # It MUST be a list of BaseMessage objects.
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# --- 4. Create the Agent and Executor ---

agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)

# `handle_parsing_errors=True` can sometimes help with unexpected LLM outputs.
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- 5. Run the Agent (Interactive Loop) ---

def run_agent():
    """Runs the grocery shopping agent in an interactive command-line loop."""
    print("Welcome to the Grocery Shopping Assistant!")
    print("I can help you create a grocery list based on your preferences.")
    print("Type 'exit' to end the conversation.")
    print("-" * 50)

    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        try:
            # When invoking, ensure agent_scratchpad is provided as an empty list
            # if there are no prior tool steps. LangChain will populate it.
            result = agent_executor.invoke(
                {"input": user_input, "chat_history": chat_history, "agent_scratchpad": []}
            )
            assistant_response = result["output"]

            print("Assistant:", assistant_response)

            # Update chat history for the next turn
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=assistant_response))

        except Exception as e:
            # Check if the error is due to agent_scratchpad type
            if "agent_scratchpad should be a list of base messages" in str(e):
                print("Internal agent error related to scratchpad format. Trying to re-initialize.")
                # This often indicates an issue in the agent's internal state.
                # For robustness, we might clear history or warn the user.
                # In this basic example, we just let the loop continue after printing the error.
            else:
                print(f"An unexpected error occurred: {e}")
            print("Please try again or type 'exit' to quit.")

if __name__ == "__main__":
    run_agent()