# Agent API Usage Examples

Practical examples for using the LangGraph-based Agent API (`/api/agent/*`).

## Prerequisites

```bash
# Start the backend server
PYTHONPATH=src poetry run uvicorn src.history_book.api.main:app --reload --port 8000

# Server will be available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

---

## cURL Examples

### Quick Start - Single Message

```bash
# Create session
SESSION_ID=$(curl -s -X POST http://localhost:8000/api/agent/sessions \
  -H "Content-Type: application/json" \
  -d '{"title": "History Questions"}' | jq -r '.id')

echo "Session ID: $SESSION_ID"

# Send message
curl -s -X POST http://localhost:8000/api/agent/sessions/$SESSION_ID/messages \
  -H "Content-Type: application/json" \
  -d '{"content": "What were the causes of World War I?"}' | jq '.message.content'

# Get conversation history
curl -s http://localhost:8000/api/agent/sessions/$SESSION_ID/messages | \
  jq '.messages[] | {role: .role, content: .content[:100]}'

# Get graph visualization
curl -s http://localhost:8000/api/agent/sessions/$SESSION_ID/graph | jq '.mermaid'

# Delete session
curl -X DELETE http://localhost:8000/api/agent/sessions/$SESSION_ID
```

### Multi-turn Conversation

```bash
# Create session
SESSION_ID=$(curl -s -X POST http://localhost:8000/api/agent/sessions \
  -H "Content-Type: application/json" \
  -d '{"title": "Roman Empire"}' | jq -r '.id')

# Question 1
echo "Q1: Who was Julius Caesar?"
curl -s -X POST http://localhost:8000/api/agent/sessions/$SESSION_ID/messages \
  -H "Content-Type: application/json" \
  -d '{"content": "Who was Julius Caesar?"}' | \
  jq -r '.message.content' | head -c 200
echo -e "\n..."

# Question 2 (uses context from Q1)
echo -e "\nQ2: When was he assassinated?"
curl -s -X POST http://localhost:8000/api/agent/sessions/$SESSION_ID/messages \
  -H "Content-Type: application/json" \
  -d '{"content": "When was he assassinated?"}' | \
  jq -r '.message.content'

# Question 3 (uses full conversation context)
echo -e "\nQ3: Who were the main conspirators?"
curl -s -X POST http://localhost:8000/api/agent/sessions/$SESSION_ID/messages \
  -H "Content-Type: application/json" \
  -d '{"content": "Who were the main conspirators?"}' | \
  jq -r '.message.content' | head -c 200
echo -e "\n..."

# View full conversation
echo -e "\n=== Full Conversation ===\"
curl -s http://localhost:8000/api/agent/sessions/$SESSION_ID/messages | \
  jq '.messages[] | "\(.role): \(.content[:80])..."'
```

### Batch Testing Multiple Queries

```bash
# Test queries array
QUERIES=(
  "What was the Renaissance?"
  "Who were the key Renaissance figures?"
  "How did it influence modern Europe?"
)

# Create session
SESSION_ID=$(curl -s -X POST http://localhost:8000/api/agent/sessions \
  -H "Content-Type: application/json" \
  -d '{"title": "Renaissance Study"}' | jq -r '.id')

# Send all queries
for query in "${QUERIES[@]}"; do
  echo -e "\n=== Q: $query ==="
  curl -s -X POST http://localhost:8000/api/agent/sessions/$SESSION_ID/messages \
    -H "Content-Type: application/json" \
    -d "{\"content\": \"$query\"}" | \
    jq -r '.message | "A: \(.content[:150])...\nCitations: \(.citations | length) sources"'
done
```

---

## Python Examples

### Using httpx (Async)

```python
import httpx
import asyncio

async def basic_chat():
    """Simple question and answer"""
    async with httpx.AsyncClient() as client:
        # Create session
        response = await client.post(
            "http://localhost:8000/api/agent/sessions",
            json={"title": "History Chat"}
        )
        session_id = response.json()["id"]
        print(f"Session ID: {session_id}")

        # Send message
        response = await client.post(
            f"http://localhost:8000/api/agent/sessions/{session_id}/messages",
            json={"content": "What was the French Revolution?"}
        )

        result = response.json()["message"]
        print(f"\nResponse: {result['content'][:300]}...")
        print(f"Citations: {len(result['citations'])} sources")
        print(f"Paragraphs: {result['metadata']['num_retrieved_paragraphs']}")

asyncio.run(basic_chat())
```

### Multi-turn Conversation

```python
async def multi_turn_conversation():
    """Demonstrate checkpointing with conversation context"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Create session
        response = await client.post(
            "http://localhost:8000/api/agent/sessions",
            json={"title": "World War I Study"}
        )
        session_id = response.json()["id"]

        # Conversation with context
        questions = [
            "What caused World War I?",
            "Which countries were involved?",  # Uses context from Q1
            "How did it end?",  # Uses context from Q1 & Q2
        ]

        for i, question in enumerate(questions, 1):
            print(f"\n{'='*60}")
            print(f"Q{i}: {question}")
            print('='*60)

            response = await client.post(
                f"http://localhost:8000/api/agent/sessions/{session_id}/messages",
                json={"content": question}
            )

            result = response.json()["message"]
            print(f"A{i}: {result['content'][:250]}...")
            print(f"\nSources: {len(result['citations'])}")

        # Get full conversation history
        response = await client.get(
            f"http://localhost:8000/api/agent/sessions/{session_id}/messages"
        )
        messages = response.json()["messages"]

        print(f"\n{'='*60}")
        print(f"CONVERSATION SUMMARY ({len(messages)} messages)")
        print('='*60)

        for msg in messages:
            role_symbol = "👤" if msg['role'] == "user" else "🤖"
            print(f"{role_symbol} {msg['content'][:80]}...")

asyncio.run(multi_turn_conversation())
```

### Session Management

```python
async def session_management():
    """List, view, and delete sessions"""
    async with httpx.AsyncClient() as client:
        # Create multiple sessions
        sessions = []
        for title in ["Ancient History", "Medieval Times", "Modern Era"]:
            response = await client.post(
                "http://localhost:8000/api/agent/sessions",
                json={"title": title}
            )
            sessions.append(response.json())
            print(f"Created: {title} ({response.json()['id']})")

        # List all sessions
        response = await client.get("http://localhost:8000/api/agent/sessions?limit=10")
        all_sessions = response.json()["sessions"]

        print(f"\nTotal sessions: {len(all_sessions)}")
        for session in all_sessions[:5]:
            print(f"  - {session.get('title', 'Untitled')}: {session['id'][:8]}...")

        # Delete sessions
        for session in sessions:
            await client.delete(
                f"http://localhost:8000/api/agent/sessions/{session['id']}"
            )
            print(f"Deleted: {session['title']}")

asyncio.run(session_management())
```

### Using Direct Service (No API)

```python
from history_book.services.graph_chat_service import GraphChatService

async def direct_service_usage():
    """Use GraphChatService directly without API"""
    service = GraphChatService()

    # Create session
    session = await service.create_session(title="Direct Service Test")
    print(f"Session: {session.id}")

    # Send messages
    questions = [
        "What was the Industrial Revolution?",
        "When did it start?",
        "What were its main impacts?"
    ]

    for question in questions:
        result = await service.send_message(
            session_id=session.id,
            user_message=question
        )

        print(f"\nQ: {question}")
        print(f"A: {result.message.content[:200]}...")
        print(f"Sources: {len(result.retrieved_paragraphs)} paragraphs")

    # Get history
    messages = await service.get_session_messages(session.id)
    print(f"\nTotal messages in session: {len(messages)}")

import asyncio
asyncio.run(direct_service_usage())
```

---

## JavaScript/TypeScript Examples

### Using fetch (Node.js or Browser)

```javascript
async function chatWithAgent() {
  const baseUrl = "http://localhost:8000";

  // Create session
  const sessionResp = await fetch(`${baseUrl}/api/agent/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: "History Chat" })
  });
  const { id: sessionId } = await sessionResp.json();
  console.log(`Session ID: ${sessionId}`);

  // Send message
  const messageResp = await fetch(
    `${baseUrl}/api/agent/sessions/${sessionId}/messages`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content: "What is ancient Egypt?" })
    }
  );

  const result = await messageResp.json();
  console.log("Response:", result.message.content.substring(0, 200) + "...");
  console.log("Citations:", result.message.citations.length);
  console.log("Paragraphs:", result.message.metadata.num_retrieved_paragraphs);

  // Get graph visualization
  const graphResp = await fetch(
    `${baseUrl}/api/agent/sessions/${sessionId}/graph`
  );
  const graph = await graphResp.json();
  console.log("\nGraph structure:");
  console.log("Nodes:", graph.nodes);
  console.log("Edges:", graph.edges);
}

chatWithAgent();
```

### Multi-turn with TypeScript

```typescript
interface Message {
  id: string;
  content: string;
  role: string;
  timestamp: string;
  session_id: string;
  citations: string[] | null;
  metadata: {
    num_retrieved_paragraphs?: number;
    graph_execution?: string;
  };
}

interface ChatResponse {
  message: Message;
}

async function multiTurnChat() {
  const baseUrl = "http://localhost:8000";

  // Create session
  const sessionResp = await fetch(`${baseUrl}/api/agent/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: "Ancient Rome" })
  });

  const { id: sessionId } = await sessionResp.json();

  // Conversation
  const questions = [
    "Who founded Rome?",
    "When was the city founded?",
    "What legends are associated with its founding?"
  ];

  for (const question of questions) {
    console.log(`\nQ: ${question}`);

    const resp = await fetch(
      `${baseUrl}/api/agent/sessions/${sessionId}/messages`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: question })
      }
    );

    const data: ChatResponse = await resp.json();
    console.log(`A: ${data.message.content.substring(0, 150)}...`);
  }

  // Get full history
  const historyResp = await fetch(
    `${baseUrl}/api/agent/sessions/${sessionId}/messages`
  );
  const { messages } = await historyResp.json();

  console.log(`\nConversation has ${messages.length} messages`);
}

multiTurnChat();
```

### React Hook Example

```typescript
import { useState, useEffect } from 'react';
import axios from 'axios';

interface Message {
  id: string;
  content: string;
  role: string;
  citations: string[] | null;
}

function useAgentChat(baseUrl: string = 'http://localhost:8000') {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Create session on mount
  useEffect(() => {
    const createSession = async () => {
      try {
        const response = await axios.post(`${baseUrl}/api/agent/sessions`, {
          title: 'Chat Session'
        });
        setSessionId(response.data.id);
      } catch (err) {
        setError('Failed to create session');
      }
    };

    createSession();
  }, [baseUrl]);

  // Load messages when session is created
  useEffect(() => {
    if (!sessionId) return;

    const loadMessages = async () => {
      try {
        const response = await axios.get(
          `${baseUrl}/api/agent/sessions/${sessionId}/messages`
        );
        setMessages(response.data.messages);
      } catch (err) {
        setError('Failed to load messages');
      }
    };

    loadMessages();
  }, [sessionId, baseUrl]);

  const sendMessage = async (content: string) => {
    if (!sessionId) {
      setError('No active session');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(
        `${baseUrl}/api/agent/sessions/${sessionId}/messages`,
        { content }
      );

      const newMessage = response.data.message;
      setMessages(prev => [...prev, newMessage]);

      return newMessage;
    } catch (err) {
      setError('Failed to send message');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const clearHistory = async () => {
    if (!sessionId) return;

    try {
      await axios.delete(`${baseUrl}/api/agent/sessions/${sessionId}`);

      // Create new session
      const response = await axios.post(`${baseUrl}/api/agent/sessions`, {
        title: 'Chat Session'
      });
      setSessionId(response.data.id);
      setMessages([]);
    } catch (err) {
      setError('Failed to clear history');
    }
  };

  return { sessionId, messages, loading, error, sendMessage, clearHistory };
}

// Usage in component
function ChatComponent() {
  const { sessionId, messages, loading, error, sendMessage, clearHistory } =
    useAgentChat();

  const [input, setInput] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    await sendMessage(input);
    setInput('');
  };

  return (
    <div>
      <div className="messages">
        {messages.map(msg => (
          <div key={msg.id} className={`message ${msg.role}`}>
            <p>{msg.content}</p>
            {msg.citations && (
              <div className="citations">
                {msg.citations.map((cite, i) => (
                  <span key={i}>{cite}</span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {loading && <div>Loading...</div>}
      {error && <div className="error">{error}</div>}

      <form onSubmit={handleSubmit}>
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Ask a question..."
        />
        <button type="submit" disabled={loading}>
          Send
        </button>
      </form>

      <button onClick={clearHistory}>Clear History</button>
    </div>
  );
}

export default ChatComponent;
```

---

## Comparison Testing

### Test Both APIs Side-by-Side

```python
import asyncio
import httpx

async def compare_apis(query: str):
    """Compare LCEL chat API vs LangGraph agent API"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test Chat API (LCEL)
        chat_session = await client.post(
            "http://localhost:8000/api/chat/sessions",
            json={"title": "LCEL Test"}
        )
        chat_sid = chat_session.json()["id"]

        chat_response = await client.post(
            f"http://localhost:8000/api/chat/sessions/{chat_sid}/messages",
            json={"content": query}
        )
        chat_result = chat_response.json()["message"]

        # Test Agent API (LangGraph)
        agent_session = await client.post(
            "http://localhost:8000/api/agent/sessions",
            json={"title": "LangGraph Test"}
        )
        agent_sid = agent_session.json()["id"]

        agent_response = await client.post(
            f"http://localhost:8000/api/agent/sessions/{agent_sid}/messages",
            json={"content": query}
        )
        agent_result = agent_response.json()["message"]

        # Compare
        print(f"Query: {query}\n")
        print("=" * 60)
        print("CHAT API (LCEL)")
        print("=" * 60)
        print(f"Response length: {len(chat_result['content'])} chars")
        print(f"Citations: {len(chat_result.get('citations', []))}")
        print(f"Content: {chat_result['content'][:200]}...\n")

        print("=" * 60)
        print("AGENT API (LangGraph)")
        print("=" * 60)
        print(f"Response length: {len(agent_result['content'])} chars")
        print(f"Citations: {len(agent_result.get('citations', []))}")
        print(f"Paragraphs: {agent_result['metadata']['num_retrieved_paragraphs']}")
        print(f"Graph: {agent_result['metadata']['graph_execution']}")
        print(f"Content: {agent_result['content'][:200]}...")

# Run comparison
asyncio.run(compare_apis("What was the significance of the Battle of Waterloo?"))
```

---

## Best Practices

### Error Handling

```python
async def robust_chat(query: str, session_id: str):
    """Chat with proper error handling"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"http://localhost:8000/api/agent/sessions/{session_id}/messages",
                json={"content": query}
            )
            response.raise_for_status()

            return response.json()["message"]

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            print("Session not found. Creating new session...")
            # Handle by creating new session
        elif e.response.status_code == 500:
            print("Server error. Please try again later.")
        raise

    except httpx.TimeoutException:
        print("Request timed out. Try a simpler query.")
        raise

    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
```

### Retrying Failed Requests

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def send_message_with_retry(session_id: str, content: str):
    """Send message with automatic retry on failure"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"http://localhost:8000/api/agent/sessions/{session_id}/messages",
            json={"content": content}
        )
        response.raise_for_status()
        return response.json()

# Will retry up to 3 times with exponential backoff
```

### Session Lifecycle Management

```python
class AgentChatSession:
    """Managed session lifecycle"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id = None

    async def __aenter__(self):
        """Create session on enter"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/agent/sessions",
                json={"title": "Auto-managed Session"}
            )
            self.session_id = response.json()["id"]
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Delete session on exit"""
        if self.session_id:
            async with httpx.AsyncClient() as client:
                await client.delete(
                    f"{self.base_url}/api/agent/sessions/{self.session_id}"
                )

    async def send(self, message: str):
        """Send message to session"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/api/agent/sessions/{self.session_id}/messages",
                json={"content": message}
            )
            return response.json()["message"]

# Usage
async def main():
    async with AgentChatSession() as session:
        response1 = await session.send("Who was Napoleon?")
        print(response1["content"][:200])

        response2 = await session.send("When did he die?")
        print(response2["content"][:200])
    # Session automatically deleted

asyncio.run(main())
```

---

## See Also

- [Agent System Documentation](/src/history_book/services/agents/CLAUDE.md) - Implementation details
- [API Documentation](/src/history_book/api/CLAUDE.md) - API reference
- [Services Documentation](/src/history_book/services/CLAUDE.md) - Service layer overview
- [Comparison Test Script](/test_langgraph_comparison.py) - Automated testing
