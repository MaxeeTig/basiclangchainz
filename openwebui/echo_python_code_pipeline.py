from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
import subprocess
import os
from io import BytesIO
import base64

class Pipeline:
    def __init__(self):
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "python_code_pipeline"
        self.name = "ECHO Python Code Pipeline"
        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    def execute_python_code(self, code):
        try:
            result = subprocess.run(
                ["python", "-c", code], capture_output=True, text=True, check=True
            )
            stdout = result.stdout.strip()
            return stdout, result.returncode
        except subprocess.CalledProcessError as e:
            return e.output.strip(), e.returncode

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

#        print(messages)
        print(user_message)

        if body.get("title", False):
            print("Title Generation")
            return "TEST Python Code Pipeline"
        else:
#            stdout, return_code = self.execute_python_code(user_message)

            # Get the current directory
            current_directory = os.getcwd()
            # Print the current directory
            print(f"Current Directory: {current_directory}")

            # Open the file elk_image.jpg from the current directory
            file_path = os.path.join(current_directory, 'elk_image.jpg')
            with open(file_path, 'rb') as file:
                # Save the image to a BytesIO object
                buffer = BytesIO()
                buffer.write(file.read())
                # Encode the image as base64
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                # Store the image base64 in the result variable
                result = image_base64

            return result
