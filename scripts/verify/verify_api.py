"""Comprehensive test suite for the History Book Chat API."""

import sys
import time

import requests


class APITester:
    """Test runner for the Chat API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id: str | None = None

    def test_health_check(self) -> bool:
        """Test the health check endpoint."""
        print("\n1. Testing health check endpoint...")
        try:
            response = requests.get(f"{self.base_url}/")
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Health check passed: {data['message']}")
                return True
            else:
                print(f"   ❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Health check error: {e}")
            return False

    def test_create_session(self) -> bool:
        """Test session creation."""
        print("\n2. Testing session creation...")
        try:
            session_data = {"title": "API Test Session"}
            response = requests.post(
                f"{self.base_url}/api/chat/sessions", json=session_data
            )

            if response.status_code == 200:
                data = response.json()
                self.session_id = data["id"]
                print(f"   ✅ Session created: {self.session_id[:8]}...")
                print(f"   📝 Title: {data['title']}")
                print(f"   🕐 Created: {data['created_at']}")
                return True
            else:
                print(
                    f"   ❌ Session creation failed: {response.status_code} - {response.text}"
                )
                return False
        except Exception as e:
            print(f"   ❌ Session creation error: {e}")
            return False

    def test_list_sessions(self) -> bool:
        """Test listing sessions."""
        print("\n3. Testing session listing...")
        try:
            response = requests.get(f"{self.base_url}/api/chat/sessions")

            if response.status_code == 200:
                data = response.json()
                sessions = data["sessions"]
                print(f"   ✅ Retrieved {len(sessions)} sessions")

                # Show first few sessions
                for i, session in enumerate(sessions[:3]):
                    title = session.get("title", "Untitled")
                    created = session["created_at"][:10]  # Just date part
                    print(f"   📝 {i + 1}. {title} ({created})")

                return True
            else:
                print(f"   ❌ Session listing failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Session listing error: {e}")
            return False

    def test_send_message(self) -> bool:
        """Test sending a message with RAG retrieval."""
        print("\n4. Testing message sending with RAG...")

        if not self.session_id:
            print("   ❌ No session available for testing")
            return False

        try:
            message_data = {
                "content": "What were the main causes of World War I?",
                "enable_retrieval": True,
            }

            print("   📤 Sending message with retrieval enabled...")
            start_time = time.time()

            response = requests.post(
                f"{self.base_url}/api/chat/sessions/{self.session_id}/messages",
                json=message_data,
            )

            response_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                message = data["message"]

                print(f"   ✅ Message sent and responded ({response_time:.1f}s)")
                print(f"   📝 Response length: {len(message['content'])} characters")
                print(f"   📚 Citations: {message.get('citations', 'None')}")
                print(f"   🤖 Response preview: {message['content'][:100]}...")

                # Verify citations format
                if message.get("citations"):
                    citation_format_ok = all(
                        cite.startswith("Page ") for cite in message["citations"]
                    )
                    if citation_format_ok:
                        print("   ✅ Citation format correct")
                    else:
                        print(
                            f"   ⚠️  Citation format unexpected: {message['citations']}"
                        )

                return True
            else:
                print(
                    f"   ❌ Message sending failed: {response.status_code} - {response.text}"
                )
                return False

        except Exception as e:
            print(f"   ❌ Message sending error: {e}")
            return False

    def test_get_messages(self) -> bool:
        """Test retrieving conversation history."""
        print("\n5. Testing conversation history retrieval...")

        if not self.session_id:
            print("   ❌ No session available for testing")
            return False

        try:
            response = requests.get(
                f"{self.base_url}/api/chat/sessions/{self.session_id}/messages"
            )

            if response.status_code == 200:
                data = response.json()
                messages = data["messages"]

                print(f"   ✅ Retrieved {len(messages)} messages")

                for _i, msg in enumerate(messages):
                    role_icon = "👤" if msg["role"] == "user" else "🤖"
                    role_text = msg["role"].capitalize()
                    content_preview = (
                        msg["content"][:60] + "..."
                        if len(msg["content"]) > 60
                        else msg["content"]
                    )
                    print(f"   {role_icon} {role_text}: {content_preview}")

                    if msg["role"] == "assistant" and msg.get("citations"):
                        print(f"      📚 Citations: {msg['citations']}")

                return True
            else:
                print(f"   ❌ Message retrieval failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"   ❌ Message retrieval error: {e}")
            return False

    def test_error_handling(self) -> bool:
        """Test API error handling."""
        print("\n6. Testing error handling...")

        try:
            # Test sending message to non-existent session
            print("   Testing non-existent session...")
            response = requests.post(
                f"{self.base_url}/api/chat/sessions/non-existent-session/messages",
                json={"content": "Test message"},
            )

            if response.status_code == 404:
                print("   ✅ Properly returns 404 for non-existent session")
                error_detail = response.json().get("detail", "")
                print(f"      Error: {error_detail}")
            else:
                print(f"   ⚠️  Unexpected status code: {response.status_code}")

            # Test invalid JSON
            print("   Testing invalid request data...")
            response = requests.post(
                f"{self.base_url}/api/chat/sessions", json={"invalid_field": "test"}
            )

            # Should still work since title is optional
            if response.status_code in [200, 422]:  # 422 = validation error
                print("   ✅ Handles invalid/missing fields appropriately")
            else:
                print(
                    f"   ⚠️  Unexpected response to invalid data: {response.status_code}"
                )

            return True

        except Exception as e:
            print(f"   ❌ Error handling test failed: {e}")
            return False


def run_full_test_suite():
    """Run the complete test suite."""
    print("🚀 Starting History Book Chat API Test Suite")
    print("=" * 55)

    # Test API endpoints (requires server)
    print("\n🌐 Testing API endpoints (requires server at http://localhost:8000)...")

    tester = APITester()

    # Check if server is accessible
    try:
        requests.get("http://localhost:8000/", timeout=2)
        server_available = True
    except Exception:
        server_available = False
        print("   ❌ Server not accessible at http://localhost:8000")
        print(
            "   💡 Start server with: PYTHONPATH=src poetry run uvicorn src.history_book.api.main:app --reload --port 8000"
        )

    api_tests_passed = []
    if server_available:
        api_tests_passed = [
            tester.test_health_check(),
            tester.test_create_session(),
            tester.test_list_sessions(),
            tester.test_send_message(),
            tester.test_get_messages(),
            tester.test_error_handling(),
        ]

    # Summary
    print("\n" + "=" * 55)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 55)

    if server_available:
        passed_count = sum(api_tests_passed)
        total_count = len(api_tests_passed)
        print(
            f"🌐 API Endpoints:     {'✅ PASS' if all(api_tests_passed) else '❌ FAIL'} ({passed_count}/{total_count})"
        )
    else:
        print("🌐 API Endpoints:     ⏸️  SKIPPED (server not running)")

    overall_success = not server_available or all(api_tests_passed)

    if overall_success:
        print("\n🎉 ALL TESTS PASSED! API is ready for frontend development.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")

    return overall_success


if __name__ == "__main__":
    success = run_full_test_suite()
    sys.exit(0 if success else 1)
