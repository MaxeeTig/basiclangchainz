"""
title: Mistral Manifold Pipe
author: ross1996
version: 0.1.0
license: MIT
"""

import os
import json
import requests
import time
from typing import List, Union, Generator, Iterator, Dict
from pydantic import BaseModel, Field

# Set DEBUG to True to enable detailed debugging output
DEBUG = True


class Pipe:
    class Valves(BaseModel):
        """Configuration for Mistral API."""
        MISTRAL_API_BASE_URL: str = Field(default="https://api.mistral.ai/v1")
        MISTRAL_API_KEY: str = Field(default="")

    def __init__(self):
        self.type = "manifold"
        self.id = "mistral"
        self.name = "mistral/"
        self.valves = self.Valves(
            **{"MISTRAL_API_KEY": os.getenv("MISTRAL_API_KEY", "")}
        )

    def _debug(self, message: str):
        """Prints debug messages if DEBUG is enabled."""
        if DEBUG:
            print(message)

    def _get_headers(self) -> Dict[str, str]:
        """Returns the headers for API requests."""
        if not self.valves.MISTRAL_API_KEY:
            raise ValueError("MISTRAL_API_KEY is not set. Please configure the environment variable.")
        return {
            "Authorization": f"Bearer {self.valves.MISTRAL_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _handle_response(self, response: requests.Response) -> dict:
        """Handles and parses API responses."""
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            self._debug(f"HTTPError: {e.response.text}")
            raise
        except ValueError as e:
            self._debug(f"Invalid JSON response: {response.text}")
            raise

    def get_mistral_models(self) -> List[Dict[str, str]]:
        """Fetches available Mistral models."""
        url = f"{self.valves.MISTRAL_API_BASE_URL}/models"
        try:
            self._debug(f"Fetching models from {url}")
            headers = self._get_headers()
            response = requests.get(url, headers=headers)
            models_data = self._handle_response(response).get("data", [])
            return [
                {"id": model.get("id", "unknown"), "name": model.get("name", "Unknown Model")}
                for model in models_data
            ]
        except Exception as e:
            self._debug(f"Failed to fetch models: {e}")
            return [{"id": "mistral", "name": str(e)}]

    def pipes(self) -> List[dict]:
        """Returns a list of available models."""
        return self.get_mistral_models()

    def pipe(self, body: dict) -> Union[str, Generator[str, None, None]]:
        """Handles a single request to the pipe."""
        try:
            model = body["model"].removeprefix("mistral.")
            messages = body["messages"]
            stream = body.get("stream", False)

            if DEBUG:
                self._debug("Incoming body:")
                self._debug(json.dumps(body, indent=2))

            if stream:
                return self.stream_response(model, messages)
            return self.get_completion(model, messages)
        except KeyError as e:
            error_msg = f"Missing required key in body: {e}"
            self._debug(error_msg)
            return f"Error: {error_msg}"
        except Exception as e:
            self._debug(f"Error in pipe method: {e}")
            return f"Error: {e}"

    def stream_response(self, model: str, messages: List[dict], retries: int = 5) -> Generator[str, None, None]:
        """Streams a response from the Mistral API, handling rate limits."""
        url = f"{self.valves.MISTRAL_API_BASE_URL}/chat/completions"
        payload = {"model": model, "messages": messages, "stream": True}

        self._debug(f"Streaming response from {url}")
        self._debug(f"Payload: {json.dumps(payload, indent=2)}")

        for attempt in range(retries):
            try:
                response = requests.post(url, json=payload, headers=self._get_headers(), stream=True)
                response.raise_for_status()

                for line in response.iter_lines():
                    if line:
                        try:
                            line_data = line.decode("utf-8").lstrip("data: ")
                            event = json.loads(line_data)

                            self._debug(f"Received stream event: {event}")

                            delta_content = event.get("choices", [{}])[0].get("delta", {}).get("content")
                            if delta_content:
                                yield delta_content

                            if event.get("choices", [{}])[0].get("finish_reason") == "stop":
                                break
                        except json.JSONDecodeError:
                            self._debug(f"Failed to decode stream line: {line}")
                            continue
                return  # Exit after successful streaming
            except requests.RequestException as e:
                if response.status_code == 429 and attempt < retries - 1:
                    wait_time = 2**attempt
                    self._debug(f"Rate limited (429). Retrying after {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    self._debug(f"Stream request failed: {e}")
                    yield f"Error: {str(e)}"

    def get_completion(self, model: str, messages: List[dict], retries: int = 3) -> str:
        """Fetches a single completion response, handling rate limits."""
        url = f"{self.valves.MISTRAL_API_BASE_URL}/chat/completions"
        payload = {"model": model, "messages": messages}

        for attempt in range(retries):
            try:
                self._debug(f"Attempt {attempt + 1}: Sending completion request to {url}")
                response = requests.post(url, json=payload, headers=self._get_headers())
                data = self._handle_response(response)
                return data["choices"][0]["message"]["content"]
            except requests.RequestException as e:
                if response.status_code == 429 and attempt < retries - 1:
                    wait_time = 2**attempt
                    self._debug(f"Rate limited (429). Retrying after {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    self._debug(f"Completion request failed: {e}")
                    return f"Error: {str(e)}"
