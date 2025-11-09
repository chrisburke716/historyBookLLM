#!/usr/bin/env python3
"""
Comparison testing: LangGraph Agent API vs LCEL Chat API

Tests the same queries on both implementations and compares:
- Retrieval results (number of paragraphs)
- Response quality
- Performance (latency)
"""

import asyncio
import time
from datetime import datetime

import httpx

BASE_URL = "http://localhost:8000"

# Test queries for comparison
TEST_QUERIES = [
    "Who was Julius Caesar?",
    "What were the main causes of World War I?",
    "Describe the French Revolution in 2-3 sentences.",
    "What was the significance of the Treaty of Versailles?",
]


async def test_chat_api(query: str) -> dict:
    """Test the existing LCEL-based chat API."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Create session
        session_resp = await client.post(
            f"{BASE_URL}/api/chat/sessions", json={"title": "LCEL Test"}
        )
        session_id = session_resp.json()["id"]

        # Send message and measure time
        start = time.time()
        message_resp = await client.post(
            f"{BASE_URL}/api/chat/sessions/{session_id}/messages",
            json={"content": query},
        )
        latency = time.time() - start

        result = message_resp.json()

        return {
            "api": "chat (LCEL)",
            "session_id": session_id,
            "query": query,
            "response": result["message"]["content"],
            "num_citations": len(result["message"].get("citations", [])),
            "latency_seconds": round(latency, 2),
            "timestamp": datetime.now().isoformat(),
        }


async def test_agent_api(query: str) -> dict:
    """Test the new LangGraph-based agent API."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Create session
        session_resp = await client.post(
            f"{BASE_URL}/api/agent/sessions", json={"title": "LangGraph Test"}
        )
        session_id = session_resp.json()["id"]

        # Send message and measure time
        start = time.time()
        message_resp = await client.post(
            f"{BASE_URL}/api/agent/sessions/{session_id}/messages",
            json={"content": query},
        )
        latency = time.time() - start

        result = message_resp.json()

        return {
            "api": "agent (LangGraph)",
            "session_id": session_id,
            "query": query,
            "response": result["message"]["content"],
            "num_citations": len(result["message"].get("citations", [])),
            "num_paragraphs": result["message"]["metadata"].get(
                "num_retrieved_paragraphs", 0
            ),
            "graph_execution": result["message"]["metadata"].get("graph_execution"),
            "latency_seconds": round(latency, 2),
            "timestamp": datetime.now().isoformat(),
        }


def print_comparison(chat_result: dict, agent_result: dict):
    """Print comparison results in a readable format."""
    print("\n" + "=" * 80)
    print(f"QUERY: {chat_result['query']}")
    print("=" * 80)

    print("\n📊 RETRIEVAL COMPARISON:")
    print(f"  Chat API (LCEL):     {chat_result['num_citations']} citations")
    print(
        f"  Agent API (LangGraph): {agent_result['num_citations']} citations, {agent_result['num_paragraphs']} paragraphs"
    )

    print("\n⚡ PERFORMANCE COMPARISON:")
    print(f"  Chat API (LCEL):     {chat_result['latency_seconds']}s")
    print(f"  Agent API (LangGraph): {agent_result['latency_seconds']}s")
    diff = agent_result["latency_seconds"] - chat_result["latency_seconds"]
    if diff > 0:
        print(f"  Difference:            +{diff:.2f}s (LangGraph slower)")
    else:
        print(f"  Difference:            {diff:.2f}s (LangGraph faster)")

    print("\n💬 CHAT API RESPONSE (first 200 chars):")
    print(f"  {chat_result['response'][:200]}...")

    print("\n🤖 AGENT API RESPONSE (first 200 chars):")
    print(f"  {agent_result['response'][:200]}...")

    print(f"\n✅ Graph Execution: {agent_result['graph_execution']}")


async def run_comparison_tests():
    """Run all comparison tests."""
    print("\n" + "🔬 LANGGRAPH vs LCEL COMPARISON TESTING " + "🔬")
    print("=" * 80)
    print(f"Testing {len(TEST_QUERIES)} queries on both APIs...")
    print("=" * 80)

    all_results = []

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n\n🧪 TEST {i}/{len(TEST_QUERIES)}")
        print("-" * 80)

        # Run both APIs
        chat_result = await test_chat_api(query)
        agent_result = await test_agent_api(query)

        # Print comparison
        print_comparison(chat_result, agent_result)

        all_results.append({"chat": chat_result, "agent": agent_result})

    # Summary statistics
    print("\n\n" + "=" * 80)
    print("📈 SUMMARY STATISTICS")
    print("=" * 80)

    avg_chat_latency = sum(r["chat"]["latency_seconds"] for r in all_results) / len(
        all_results
    )
    avg_agent_latency = sum(r["agent"]["latency_seconds"] for r in all_results) / len(
        all_results
    )
    avg_chat_citations = sum(r["chat"]["num_citations"] for r in all_results) / len(
        all_results
    )
    avg_agent_citations = sum(r["agent"]["num_citations"] for r in all_results) / len(
        all_results
    )

    print("\nAverage Latency:")
    print(f"  Chat API (LCEL):     {avg_chat_latency:.2f}s")
    print(f"  Agent API (LangGraph): {avg_agent_latency:.2f}s")
    print(f"  Difference:            {avg_agent_latency - avg_chat_latency:.2f}s")

    print("\nAverage Citations:")
    print(f"  Chat API (LCEL):     {avg_chat_citations:.1f}")
    print(f"  Agent API (LangGraph): {avg_agent_citations:.1f}")

    print("\n" + "=" * 80)
    print("✅ COMPARISON TESTING COMPLETE")
    print("=" * 80)

    return all_results


if __name__ == "__main__":
    print("\n🚀 Starting comparison tests...")
    print("📝 Make sure the server is running on http://localhost:8000\n")

    results = asyncio.run(run_comparison_tests())
