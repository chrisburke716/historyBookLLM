/**
 * ChatThreadController — runs inside `<CopilotProvider>` and owns two effects:
 *
 * 1. Hydrate the CopilotKit thread with historical user/assistant messages
 *    from Weaviate when the session changes. Tool messages from prior turns
 *    are not surfaced (Weaviate stores user/assistant only).
 *
 * 2. Fire `onRunEnd` when the agent run finishes, so the parent can refresh
 *    the session list (title regeneration happens server-side post-run).
 *
 * Renders nothing.
 */

import React, { useEffect, useRef } from 'react';
import { useCoAgent, useCopilotChatInternal } from '@copilotkit/react-core';

import { MessageResponse } from '../types';

const RUN_AGENT_NAME = 'rag';

interface Props {
  historicalMessages: MessageResponse[];
  onRunEnd?: () => void;
}

const ChatThreadController: React.FC<Props> = ({ historicalMessages, onRunEnd }) => {
  const { setMessages } = useCopilotChatInternal();
  const { running } = useCoAgent({ name: RUN_AGENT_NAME });

  // Hydrate once per mount. `historicalMessages` reference can change for
  // unrelated reasons (e.g. session-list refresh), but we must not call
  // setMessages mid-conversation — that would clobber the live thread.
  //
  // The ref isn't reset within a component instance. It's reset because
  // `<CopilotKit key={threadId}>` in CopilotProvider re-keys on session
  // switch, which unmounts and remounts this entire subtree with a fresh
  // ref. One mount per session ⇒ one hydration per session.
  const hydrated = useRef(false);
  useEffect(() => {
    if (hydrated.current) return;
    hydrated.current = true;
    const seeded = historicalMessages
      .filter((m) => m.role === 'user' || m.role === 'assistant')
      .map((m) => ({
        id: m.id,
        role: m.role as 'user' | 'assistant',
        content: m.content,
      }));
    setMessages(seeded as any);
  }, [historicalMessages, setMessages]);

  // Fire onRunEnd on running: true → false transition.
  const wasRunning = useRef(false);
  useEffect(() => {
    if (wasRunning.current && !running) {
      onRunEnd?.();
    }
    wasRunning.current = running;
  }, [running, onRunEnd]);

  return null;
};

export default ChatThreadController;
