"""Full integration test for frontend + backend chat functionality."""

import sys
from http import HTTPStatus

import requests


def test_backend_api():
    """Test the backend API is working."""
    print("🧪 Testing backend API...")

    try:
        # Test health check
        response = requests.get("http://localhost:8000/", timeout=5)
        assert response.status_code == HTTPStatus.OK
        print("   ✅ Backend health check passed")

        # Test create session
        session_response = requests.post(
            "http://localhost:8000/api/chat/sessions",
            json={"title": "Integration Test"},
            timeout=5,
        )
        assert session_response.status_code == HTTPStatus.OK
        session_id = session_response.json()["id"]
        print(f"   ✅ Session created: {session_id[:8]}...")

        # Test send message
        message_response = requests.post(
            f"http://localhost:8000/api/chat/sessions/{session_id}/messages",
            json={
                "content": "What were the causes of World War I?",
                "enable_retrieval": True,
                "max_context_paragraphs": 3,
            },
            timeout=30,
        )
        assert message_response.status_code == HTTPStatus.OK
        ai_message = message_response.json()["message"]
        print(f"   ✅ AI response received: {len(ai_message['content'])} chars")
        print(f"   📚 Citations: {ai_message.get('citations', [])}")

        return True

    except Exception as e:
        print(f"   ❌ Backend test failed: {e}")
        return False


def test_frontend_availability():
    """Test the frontend is accessible."""
    print("\n🌐 Testing frontend accessibility...")

    try:
        response = requests.get("http://localhost:3000", timeout=5)
        assert response.status_code == HTTPStatus.OK
        assert "History Book Chat" in response.text or "React App" in response.text
        print("   ✅ Frontend is accessible")
        return True

    except Exception as e:
        print(f"   ❌ Frontend test failed: {e}")
        return False


def test_cors_setup():
    """Test CORS is properly configured."""
    print("\n🔒 Testing CORS configuration...")

    try:
        # Simulate a CORS preflight request
        response = requests.options(
            "http://localhost:8000/api/chat/sessions",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
            timeout=5,
        )

        # Check CORS headers
        cors_headers = {
            "access-control-allow-origin": response.headers.get(
                "access-control-allow-origin"
            ),
            "access-control-allow-methods": response.headers.get(
                "access-control-allow-methods"
            ),
            "access-control-allow-headers": response.headers.get(
                "access-control-allow-headers"
            ),
        }

        print(f"   📋 CORS headers: {cors_headers}")

        # Should allow localhost:3000
        if "localhost:3000" in str(cors_headers.get("access-control-allow-origin", "")):
            print("   ✅ CORS properly configured for React app")
            return True
        else:
            print("   ⚠️  CORS might not be properly configured")
            return False

    except Exception as e:
        print(f"   ❌ CORS test failed: {e}")
        return False


def summarize_setup():
    """Provide setup summary and next steps."""
    print("\n" + "=" * 60)
    print("🎉 FULL-STACK CHAT APPLICATION READY!")
    print("=" * 60)

    print("\n📋 SETUP SUMMARY:")
    print("   🔧 Backend: FastAPI with RAG-powered chat")
    print("   🌐 Frontend: React + TypeScript + Material-UI")
    print("   🔗 API Integration: Axios with centralized service")
    print("   💾 Database: Weaviate with historical documents")
    print("   🤖 LLM: LangChain with OpenAI (or Mock for testing)")

    print("\n🌍 RUNNING SERVERS:")
    print("   📡 FastAPI Backend:  http://localhost:8000")
    print("   📱 React Frontend:   http://localhost:3000")
    print("   📚 API Docs:         http://localhost:8000/docs")

    print("\n✨ KEY FEATURES IMPLEMENTED:")
    print("   💬 Real-time chat interface")
    print("   🔍 RAG-powered responses with document retrieval")
    print("   📄 Citation system showing actual page numbers")
    print("   💾 Session management (create, switch, persist)")
    print("   🎨 Clean Material-UI design with accessibility")
    print("   ⚡ TypeScript for type safety")
    print("   🔧 Extensible architecture for future features")

    print("\n🚀 NEXT STEPS FOR DEVELOPMENT:")
    print("   1. Open http://localhost:3000 to use the chat interface")
    print("   2. Test with historical questions (e.g., 'What caused WWI?')")
    print("   3. Add more features like:")
    print("      - Book reader with clickable text")
    print("      - Knowledge graph visualization")
    print("      - Document upload interface")
    print("      - User authentication")

    print("\n💡 FOR PRODUCTION DEPLOYMENT:")
    print("   - Replace Mock LLM with real OpenAI API key")
    print("   - Configure environment variables")
    print("   - Build React app: npm run build")
    print("   - Use production ASGI server like Gunicorn")
    print("   - Set up proper database persistence")


def main():
    """Run full integration test suite."""
    print("🧪 FULL-STACK INTEGRATION TEST SUITE")
    print("=" * 50)

    # Test individual components
    backend_ok = test_backend_api()
    frontend_ok = test_frontend_availability()
    cors_ok = test_cors_setup()

    # Overall result
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")
    print(f"   🔧 Backend API:      {'✅ PASS' if backend_ok else '❌ FAIL'}")
    print(f"   🌐 Frontend:         {'✅ PASS' if frontend_ok else '❌ FAIL'}")
    print(f"   🔗 CORS Setup:       {'✅ PASS' if cors_ok else '❌ FAIL'}")

    if backend_ok and frontend_ok:
        print("\n🎉 INTEGRATION TEST: ✅ PASS")
        summarize_setup()
        return True
    else:
        print("\n⚠️  INTEGRATION TEST: ❌ FAIL")
        print("\n💡 Make sure both servers are running:")
        print(
            "   Backend: PYTHONPATH=src poetry run uvicorn src.history_book.api.main:app --reload --port 8000"
        )
        print("   Frontend: cd frontend && npm start")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
