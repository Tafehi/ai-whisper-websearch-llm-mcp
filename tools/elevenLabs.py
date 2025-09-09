
from elevenlabs.client import ElevenLabs
from elevenlabs import stream


client = ElevenLabs()
audio_stream = client.text_to_speech.stream(
    text="Streaming test",
    voice_id="JBFqnCBsd6RMkjVDRZzb",
    model_id="eleven_multilingual_v2",
)
stream(audio_stream)



# import os
# from langchain_community.tools import ElevenLabsText2SpeechTool
# from dotenv import load_dotenv
#
# class ElevenLabsReader:
#     def __init__(self, voice_id: str = None):
#         # Ensure API key is set
#         load_dotenv()
#         if not os.getenv("ELEVENLABS_API_KEY"):
#             raise RuntimeError("Set ELEVENLABS_API_KEY in your environment.")
#
#         # # Default voice from ElevenLabs docs if none provided
#         # self.voice_id = voice_id or "JBFqnCBsd6RMkjVDRZzb"
#         #
#         # # Initialize LangChain ElevenLabs tool
#         # self.tts_tool = ElevenLabsText2SpeechTool(voice=self.voice_id)
#
#     def speak(self, text: str, stream: bool = True):
#         """
#         Speak the given text.
#         If stream=True, plays audio directly.
#         If stream=False, saves to a file and plays it.
#         """
#
#         print("[INFO] Streaming speech...")
#         tts = ElevenLabsText2SpeechTool()
#         tts.name
#         speech_file = tts.run(text)
#         tts.play(speech_file)
#         # self.tts_tool.stream_speech(text)
#
#
#
# if __name__ == "__main__":
#     reader = ElevenLabsReader()
#     reader.speak("Hello world! I am the real slim shady", stream=True)
