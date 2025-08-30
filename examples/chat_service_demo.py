"""Example usage of the ChatService for conversational historical document interaction."""

import asyncio
import logging

from src.history_book.llm import LangChainProvider, LLMConfig, MockLLMProvider
from src.history_book.services import ChatService

# Configure logging - suppress external modules, show only history_book
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger("src.history_book").setLevel(logging.INFO)

logger = logging.getLogger(__name__)


async def demonstrate_chat_service():
    """Demonstrate the full capabilities of the ChatService."""

    print("🚀 ChatService Demonstration")
    print("=" * 50)

    # Initialize the chat service with mock provider for demonstration
    llm_config = LLMConfig.from_environment()  # Load from environment variables
    llm_provider = LangChainProvider(llm_config)
    chat_service = ChatService(llm_provider=llm_provider)

    try:
        # 1. Create a new chat session
        print("\n📝 Creating a new chat session...")
        session = await chat_service.create_session("World War I Discussion")
        print(f"   ✅ Created session: {session.id}")
        print(f"   📅 Created at: {session.created_at}")
        print(f"   📌 Title: {session.title}")

        # 2. Send a series of messages to build conversation
        questions = [
            "What were the main causes of World War I?",
            "How did the alliance system contribute to the conflict?",
            "What role did the assassination of Archduke Franz Ferdinand play?",
            "How did the war end and what were its consequences?",
        ]

        print("\n💬 Starting conversation...")
        for i, question in enumerate(questions, 1):
            print(f"\n🤔 Question {i}: {question}")

            # Send message and get response
            response = await chat_service.send_message(
                session_id=session.id, user_message=question, enable_retrieval=True
            )

            print(f"🤖 AI Response: {response.content[:100]}...")
            print(f"   📊 Response length: {len(response.content)} characters")
            if response.retrieved_paragraphs:
                print(f"   📚 Context paragraphs: {len(response.retrieved_paragraphs)}")

        # 3. Demonstrate streaming response
        print("\n🌊 Demonstrating streaming response...")
        print("🤔 Question: Tell me more about the Treaty of Versailles")
        print("🤖 Streaming Response: ", end="")

        chunks = []
        async for chunk in chat_service.send_message_stream(
            session_id=session.id,
            user_message="Tell me more about the Treaty of Versailles",
            enable_retrieval=True,
        ):
            chunks.append(chunk)
            print(chunk, end="", flush=True)

        print(f"\n   📊 Streamed {len(chunks)} chunks")

        # 4. Review conversation history
        print("\n📜 Conversation history...")
        messages = await chat_service.get_session_messages(session.id)
        print(f"   💬 Total messages in session: {len(messages)}")

        for i, msg in enumerate(messages[-4:]):  # Show last 4 messages
            role_icon = "🤔" if str(msg.role) == "user" else "🤖"
            role_text = str(msg.role).capitalize()
            print(f"   {role_icon} {role_text}: {msg.content[:60]}...")

        # 5. Demonstrate session management
        print("\n📁 Session management...")
        recent_sessions = await chat_service.list_recent_sessions(3)
        print(f"   📈 Recent sessions: {len(recent_sessions)}")

        for session_info in recent_sessions:
            print(f"   📝 {session_info.title} (ID: {session_info.id[:8]}...)")
            print(f"      🕒 Updated: {session_info.updated_at}")

        # 6. Demonstrate message search (if implemented)
        print("\n🔍 Message search demonstration...")
        search_results = await chat_service.search_messages(
            query="alliance system", session_id=session.id, limit=3
        )
        print(
            f"   🎯 Found {len(search_results)} messages mentioning 'alliance system'"
        )

        # 7. Show session statistics
        print("\n📊 Session statistics...")
        all_messages = await chat_service.get_session_messages(session.id)
        user_messages = [m for m in all_messages if str(m.role) == "user"]
        ai_messages = [m for m in all_messages if str(m.role) == "assistant"]

        print(f"   👤 User messages: {len(user_messages)}")
        print(f"   🤖 AI messages: {len(ai_messages)}")

        total_chars = sum(len(m.content) for m in all_messages)
        print(f"   📝 Total conversation length: {total_chars} characters")

        if ai_messages:
            avg_response_length = sum(len(m.content) for m in ai_messages) / len(
                ai_messages
            )
            print(
                f"   📏 Average AI response length: {avg_response_length:.0f} characters"
            )

        print("\n✨ ChatService demonstration completed successfully!")

    except Exception as e:
        logger.error(f"Error during demonstration: {e}")
        raise
    finally:
        # Always clean up
        chat_service.close()
        print("\n🧹 ChatService closed and cleaned up")


async def demonstrate_error_handling():
    """Demonstrate error handling in ChatService."""

    print("\n🚨 Error handling demonstration")
    print("-" * 30)

    chat_service = ChatService(llm_provider=MockLLMProvider(LLMConfig()))

    try:
        # Try to get a non-existent session
        print("🔍 Attempting to get non-existent session...")
        non_existent = await chat_service.get_session("non-existent-id")
        print(f"   Result: {non_existent}")  # Should be None

        # Try to send message to non-existent session (this will create the message but might error)
        print("💬 Attempting to send message to non-existent session...")
        try:
            response = await chat_service.send_message(
                session_id="non-existent-session",
                user_message="Test message",
                enable_retrieval=False,
            )
            print(f"   Unexpectedly succeeded: {response.content[:50]}...")
        except Exception as e:
            print(f"   ✅ Properly caught error: {type(e).__name__}: {e}")

        print("✅ Error handling demonstration completed")

    finally:
        chat_service.close()


if __name__ == "__main__":
    print("🎭 ChatService Example & Demonstration")
    print("=" * 60)

    # Run the main demonstration
    asyncio.run(demonstrate_chat_service())

    # Run error handling demonstration
    asyncio.run(demonstrate_error_handling())

    print("\n🎉 All demonstrations completed!")
    print("\n💡 Next steps:")
    print("   - Install LangChain dependencies for production LLM providers")
    print("   - Load historical documents into the database")
    print("   - Enable retrieval for context-aware responses")
    print("   - Build a user interface (CLI, web, or API)")
    print("   - Configure production LLM provider (OpenAI, Anthropic, etc.)")
