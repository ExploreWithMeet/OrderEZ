from os import getenv
from dotenv import load_dotenv

load_dotenv()


class Config:
    convex_url: str = getenv("CONVEX_URL")
