# import os

# from langchain_community.tools import ElevenLabsText2SpeechTool
#
 # picks up ELEVENLABS_API_KEY from .env

#
# ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY"),
#
#
#
# text_to_speak = "Hello world! I am the real slim shady"
#
# tts = ElevenLabsText2SpeechTool()
# tts.name
# tts.stream_speech(text_to_speak)
from elevenlabs.client import ElevenLabs
import os
from dotenv import load_dotenv

load_dotenv()

client = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY")
)

response = client.conversational_ai.agents.create(
    name="My conversational agent",
    conversation_config={
        "agent": {
            "prompt": {
                "prompt": "You are a helpful assistant that can answer questions and help with tasks.",
            }
        }
    }
)

