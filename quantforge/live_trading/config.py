import os
from dataclasses import dataclass


@dataclass
class KiteConfig:
    api_key: str
    access_token: str

    @classmethod
    def from_env(cls):
        return cls(
            api_key=os.environ["KITE_API_KEY"],
            access_token=os.environ["KITE_ACCESS_TOKEN"],
        )
