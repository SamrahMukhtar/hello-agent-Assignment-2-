# Import necessary libraries
import asyncio
import os  # For accessing environment variables

# Import modules from a custom 'agents' library
from agents import (
    Agent,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    Runner,
    set_default_openai_client,
    set_tracing_disabled,
)
from dotenv import load_dotenv  # To load environment variables from a .env file

# Load environment variables from .env file into the environment
load_dotenv()

# Get the Gemini API key from the environment variables
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Initialize an asynchronous OpenAI-style client with the Gemini API key and URL
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Set the external client as the default OpenAI client
set_default_openai_client(external_client)

# Disable tracing/logging for this run
set_tracing_disabled(True)
# Define the chat model to use with the agent
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash", openai_client=external_client
)


async def Healthcare():
    # Create an agent with a name, instructions, and the model to use
    Healthcare_agent = Agent(
        name="HealthMate",
        instructions="""Always remind the user that you are not a licensed healthcare professional.
        Avoid providing medical diagnoses or recommending treatments.
        If a question is urgent or serious, advise the user to consult a doctor.
        Offer tips on general wellness, such as hydration, sleep, and nutrition.
        Use plain, friendly language suitable for all audiences.
        When unsure about something, express uncertainty instead of guessing.
        Do not provide information about prescription medications or dosages.""",
        model=model,
    )

    prompt_input = input("I am your Healthmate, Ask Anything:")
    output = await Runner.run(Healthcare_agent, prompt_input)
    print("Output: " + output.final_output)

    file = open("README.md", "a")
    file.write("#Prompt:\n")
    file.write(f"{prompt_input}\n\n")

    file.write("#Output:\n")
    file.write(f"{output.final_output}\n")


def run_Healthcare():
    asyncio.run(Healthcare())
