import os
from pathlib import Path
from dotenv import load_dotenv
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, BotoCoreError

class PollyTTS:
    def __init__(self):
        """
        Initialize Polly once. Provide voice/region via params or env.
        ENV:
          - AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN (optional)
          - AWS_REGION (default: eu-west-1)
          - POLLY_VOICE_ID (default: Joanna)
        """
        load_dotenv()

        self.voice_id = "Joanna"

        try:
            self.polly = boto3.client(
                "polly",
                region_name=os.getenv("AWS_REGION"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
            )
        except (NoCredentialsError, PartialCredentialsError):
            raise RuntimeError("AWS credentials are missing or incomplete.")
        except BotoCoreError as e:
            raise RuntimeError(f"Error initializing AWS client: {e}")

    def speak(self, text: str, audio_type: str, out_dir: str = "./tools") -> str:
        """
        Convert text to speech and save as ./tools/{audio_type}.mp3
        :param text: The text to synthesize
        :param audio_type: 'question' or 'answer' (used in filename)
        :param out_dir: Output directory (default ./tools)
        :return: Path to saved mp3
        """
        if not text:
            raise ValueError("text must be non-empty")

        audio_type = audio_type.lower().strip()
        if audio_type not in {"question", "answer"}:
            raise ValueError("audio_type must be 'question' or 'answer'.")

        try:
            resp = self.polly.synthesize_speech(
                Text=text,
                OutputFormat="mp3",
                VoiceId=self.voice_id,
            )
        except BotoCoreError as e:
            raise RuntimeError(f"Polly error: {e}")

        audio = resp.get("AudioStream")
        if not audio:
            raise RuntimeError("Polly did not return an AudioStream.")

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        file_path = os.path.join(out_dir, f"{audio_type}.mp3")

        with open(file_path, "wb") as f:
            f.write(audio.read())

        print(f"Audio saved to {file_path}")
        return file_path
