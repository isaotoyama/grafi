import asyncio
import os
import uuid

from grafi.common.containers.container import container
from grafi.common.events.topic_events.consume_from_topic_event import (
    ConsumeFromTopicEvent,
)
from grafi.common.models.execution_context import ExecutionContext
from grafi.common.models.message import Message
from grafi.nodes.impl.llm_node import LLMNode
from grafi.tools.llms.impl.openai_tool import OpenAITool
from grafi.tools.llms.llm_stream_response_command import LLMStreamResponseCommand


event_store = container.event_store

api_key = os.getenv("OPENAI_API_KEY")


def get_execution_context():
    return ExecutionContext(
        conversation_id="conversation_id",
        execution_id=uuid.uuid4().hex,
        assistant_request_id=uuid.uuid4().hex,
    )


def test_openai_tool_stream():
    event_store.clear_events()
    openai_tool = OpenAITool.Builder().api_key(api_key).build()
    content = ""
    for message in openai_tool.stream(
        get_execution_context(),
        [Message(role="user", content="Hello, my name is Grafi, how are you doing?")],
    ):
        assert message.role == "assistant"
        if message.content is not None:
            content += message.content
            print(message.content, end="", flush=True)

    assert len(event_store.get_events()) == 2
    assert content is not None
    assert "Grafi" in content


async def test_openai_tool_a_stream():
    event_store.clear_events()
    openai_tool = OpenAITool.Builder().api_key(api_key).build()
    content = ""
    async for message in openai_tool.a_stream(
        get_execution_context(),
        [Message(role="user", content="Hello, my name is Grafi, how are you doing?")],
    ):
        assert message.role == "assistant"
        if message.content is not None:
            content += message.content
            print(message.content + "_", end="", flush=True)

    assert len(event_store.get_events()) == 2
    assert content is not None
    assert "Grafi" in content


def test_openai_tool():
    openai_tool = OpenAITool.Builder().api_key(api_key).build()
    event_store.clear_events()
    message = openai_tool.execute(
        get_execution_context(),
        [Message(role="user", content="Hello, my name is Grafi, how are you doing?")],
    )

    assert message.role == "assistant"

    print(message.content)

    assert len(event_store.get_events()) == 2
    assert message.content is not None
    assert "Grafi" in message.content


def test_openai_tool_with_chat_param():
    chat_param = {
        "temperature": 0.1,
        "max_tokens": 15,
    }
    openai_tool = OpenAITool.Builder().api_key(api_key).chat_params(chat_param).build()
    event_store.clear_events()
    message = openai_tool.execute(
        get_execution_context(),
        [Message(role="user", content="Hello, my name is Grafi, how are you doing?")],
    )

    assert message.role == "assistant"

    print(message.content)

    assert len(event_store.get_events()) == 2
    assert message.content is not None
    assert "Grafi" in message.content
    assert len(message.content) < 70


async def test_openai_tool_async():
    openai_tool = OpenAITool.Builder().api_key(api_key).build()
    event_store.clear_events()

    content = ""
    async for message in openai_tool.a_execute(
        get_execution_context(),
        [Message(role="user", content="Hello, my name is Grafi, how are you doing?")],
    ):
        assert message.role == "assistant"
        if message.content is not None:
            content += message.content

    print(content)

    assert "Grafi" in content

    print(len(event_store.get_events()))

    assert len(event_store.get_events()) == 2


async def test_llm_a_stream_node():
    event_store.clear_events()
    llm_stream_node = (
        LLMNode.Builder()
        .command(
            LLMStreamResponseCommand.Builder()
            .llm(OpenAITool.Builder().api_key(api_key).build())
            .build()
        )
        .build()
    )

    content = ""

    execution_context = get_execution_context()

    topic_event = ConsumeFromTopicEvent(
        execution_context=execution_context,
        topic_name="test_topic",
        consumer_name="LLMNode",
        consumer_type="LLMNode",
        offset=-1,
        data=[
            Message(role="user", content="Hello, my name is Grafi, how are you doing?")
        ],
    )

    async for message in llm_stream_node.a_execute(
        execution_context,
        [topic_event],
    ):
        assert message.role == "assistant"
        if message.content is not None:
            content += message.content
            print(message.content, end="", flush=True)

    assert content is not None
    assert "Grafi" in content
    assert len(event_store.get_events()) == 4


test_openai_tool()
test_openai_tool_with_chat_param()
test_openai_tool_stream()
asyncio.run(test_openai_tool_a_stream())
asyncio.run(test_openai_tool_async())
asyncio.run(test_llm_a_stream_node())