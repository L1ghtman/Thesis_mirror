from typing import Tuple

# Llama-3 prompt tokens
BOS = "<|begin_of_text|>"
EOS = "<|end_of_text|>"
EOT = "<|eot_id|>"

# These wrap around one of the three possible roles: system, user, assistant
SH = "<|start_header_id|>"
EH = "<|end_header_id|>"
SP = "You are a helpful AI assistant for travel tips and recommendations"

# Take a prompt and format it into Llama-3 prompt token format according to https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
def prompt_format(prompt: str, first_call: bool) -> Tuple[str, bool]:
    """Take a user input and format it into llama3 prompt token format."""
    if first_call:
        first_call = False
        # TODO: insert '\n\n' into tokens itself
        return (f"{BOS}{SH}system{EH}\n\n{SP}{EOT}{SH}user{EH}\n\n{prompt}{EOT}{SH}assistant{EH}\n\n", first_call)
    else:
        return (f"{prompt}{EOT}{SH}assistant{EH}\n\n", first_call)