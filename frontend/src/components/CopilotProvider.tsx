/**
 * CopilotProvider — wraps the chat surface with `<CopilotKit>` configured to
 * talk to our self-hosted AG-UI endpoint via `@ag-ui/client`'s HttpAgent.
 *
 * Scoped to the chat page only — do not move higher in the tree.
 */

import React, { useMemo } from 'react';
import { CopilotKit } from '@copilotkit/react-core';
import { HttpAgent } from '@ag-ui/client';

const AGENT_NAME = 'rag';
const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const AGENT_URL = `${API_BASE}/api/chat/agent`;

// CopilotKit requires a `runtimeUrl` (or cloud key) for prop validation, and
// its internal runtime client will probe that URL for thread/runtime endpoints
// even when `selfManagedAgents` is set. Pointing it at a dedicated stub keeps
// those probes off the real agent endpoint.
const RUNTIME_STUB_URL = `${API_BASE}/api/chat/copilotkit-stub`;

interface Props {
  threadId: string;
  children: React.ReactNode;
}

const CopilotProvider: React.FC<Props> = ({ threadId, children }) => {
  // Construct the HttpAgent with the session's threadId so it round-trips to
  // the backend as RunAgentInput.threadId — that's what our route resolves to
  // session_id.
  const agent = useMemo(
    () => new HttpAgent({ url: AGENT_URL, threadId }),
    [threadId],
  );

  // Re-key on threadId so CopilotKit's runtime resets cleanly when the user
  // switches sessions (history is hydrated separately by ChatThreadController).
  return (
    <CopilotKit
      key={threadId}
      runtimeUrl={RUNTIME_STUB_URL}
      selfManagedAgents={{ [AGENT_NAME]: agent }}
      agent={AGENT_NAME}
      threadId={threadId}
      showDevConsole={false}
    >
      {children}
    </CopilotKit>
  );
};

export default CopilotProvider;
