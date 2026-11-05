import os
from dotenv import load_dotenv


def load_secrets():
    """Load secrets from .env file."""
    load_dotenv("../../.env")
    return os.environ["JOPTIM_API_KEY"], os.environ["JOPTIM_SERVER_URL"]
