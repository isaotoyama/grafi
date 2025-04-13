import json
import uuid

from kyc_assistant import KycAssistant
from grafi.common.decorators.llm_function import llm_function
from grafi.common.models.execution_context import ExecutionContext
from grafi.common.models.message import Message
from grafi.tools.functions.function_tool import FunctionTool


class ClientInfo(FunctionTool):

    @llm_function
    def request_client_information(self, question_description: str):
        """
        Requests client input for personal information based on a given question description.
        """
        return json.dumps({"question_description": question_description})


class RegisterClient(FunctionTool):

    @llm_function
    def register_client(self, name: str, email: str):
        """
        Registers a user based on their name and email.
        """
        return f"User {name}, email {email} has been registered."


user_info_extract_system_message = """
"You are a strict validator designed to check whether a given input contains a user's full name and email address. Your task is to analyze the input and determine if both a full name (first and last name) and a valid email address are present.

### Validation Criteria:
- **Full Name**: The input should contain at least two words that resemble a first and last name. Ignore common placeholders (e.g., 'John Doe').
- **Email Address**: The input should include a valid email format (e.g., example@domain.com).
- **Case Insensitivity**: Email validation should be case insensitive.
- **Accuracy**: Avoid false positives by ensuring random text, usernames, or partial names don’t trigger validation.
- **Output**: Respond with Valid if both a full name and an email are present, otherwise respond with Invalid. Optionally, provide a reason why the input is invalid.

### Example Responses:
- **Input**: "John Smith, john.smith@email.com" → **Output**: "Valid"
- **Input**: "john.smith@email.com" → **Output**: "Invalid - Full name is missing"
- **Input**: "John" → **Output**: "Invalid - Full name and email are missing"

Strictly follow these validation rules and do not assume missing details."
"""


def get_execution_context():
    return ExecutionContext(
        conversation_id="conversation_id",
        execution_id=uuid.uuid4().hex,
        assistant_request_id=uuid.uuid4().hex,
    )


def test_kyc_assistant():
    execution_context = get_execution_context()

    assistant = (
        KycAssistant.Builder()
        .name("KycAssistant")
        .api_key(
            "YOUR_OPENAI_API_KEY"
        )
        .user_info_extract_system_message(user_info_extract_system_message)
        .action_llm_system_message(
            "Select the most appropriate tool based on the request."
        )
        .summary_llm_system_message(
            "Response to user with result of registering. You must include 'registered' in the response if succeed."
        )
        .hitl_request(ClientInfo(name="request_human_information"))
        .register_request(RegisterClient(name="register_client"))
        .build()
    )

    while True:
        # Initial User Input
        user_input = input("User: ")
        input_data = [Message(role="user", content=user_input)]

        output = assistant.execute(execution_context, input_data)

        responses = []
        for message in output:
            try:
                content_json = json.loads(message.content)
                responses.append(content_json["question_description"])
            except json.JSONDecodeError:
                responses.append(message.content)

        respond_to_user = " and ".join(responses)
        print("Assistant:", respond_to_user)

        if "registered" in output[0].content:
            break


if __name__ == "__main__":
    test_kyc_assistant()