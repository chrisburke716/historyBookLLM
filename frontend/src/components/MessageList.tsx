/**
 * MessageList — renders the live conversation from CopilotKit hooks.
 *
 * Subscribes to:
 *  - `useCopilotChatInternal()` for the live message thread (AG-UI shape)
 *  - `useCoAgent({ name: 'rag' })` for agent state (retrieved_paragraphs)
 *
 * Tool-call cards are live-only — they do not persist on reload (same
 * behavior as the original plan).
 */

import React, { useMemo } from 'react';
import {
  Avatar,
  Box,
  Chip,
  List,
  ListItem,
  Paper,
  Typography,
} from '@mui/material';
import {
  Person as PersonIcon,
  SmartToy as BotIcon,
} from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useCoAgent, useCopilotChatInternal } from '@copilotkit/react-core';

import ToolCallCard from './ToolCallCard';

interface ParagraphState {
  book_index: number;
  chapter_index: number;
  page: number;
  paragraph_index?: number;
  id?: string;
  text?: string;
}

interface AgentState {
  retrieved_paragraphs?: ParagraphState[];
}

const RUN_AGENT_NAME = 'rag';

const MAX_ARG_PREVIEW_LEN = 60;
const MAX_RESULT_PREVIEW_LEN = 240;

function truncate(s: string, n: number): string {
  return s.length > n ? s.slice(0, n - 1) + '…' : s;
}

function argsPreview(args: unknown): string | undefined {
  if (!args) return undefined;
  if (typeof args === 'string') {
    return truncate(args, MAX_ARG_PREVIEW_LEN);
  }
  try {
    return truncate(JSON.stringify(args), MAX_ARG_PREVIEW_LEN);
  } catch {
    return undefined;
  }
}

function citationLabel(p: ParagraphState): string {
  return `[B${p.book_index}, Ch${p.chapter_index}, p.${p.page}]`;
}

const markdownComponents = {
  // Keep markdown inside the bubble — no extra outer margins.
  p: ({ children }: { children?: React.ReactNode }) => (
    <Typography variant="body1" sx={{ my: 0.5 }}>{children}</Typography>
  ),
  h1: ({ children }: { children?: React.ReactNode }) => (
    <Typography variant="h6" sx={{ mt: 1, mb: 0.5, fontWeight: 700 }}>{children}</Typography>
  ),
  h2: ({ children }: { children?: React.ReactNode }) => (
    <Typography variant="subtitle1" sx={{ mt: 1, mb: 0.5, fontWeight: 700 }}>{children}</Typography>
  ),
  h3: ({ children }: { children?: React.ReactNode }) => (
    <Typography variant="subtitle2" sx={{ mt: 1, mb: 0.5, fontWeight: 700 }}>{children}</Typography>
  ),
  ul: ({ children }: { children?: React.ReactNode }) => (
    <Box component="ul" sx={{ pl: 3, my: 0.5 }}>{children}</Box>
  ),
  ol: ({ children }: { children?: React.ReactNode }) => (
    <Box component="ol" sx={{ pl: 3, my: 0.5 }}>{children}</Box>
  ),
  li: ({ children }: { children?: React.ReactNode }) => (
    <Box component="li" sx={{ my: 0.25 }}>{children}</Box>
  ),
  code: ({ children }: { children?: React.ReactNode }) => (
    <Box
      component="code"
      sx={{
        bgcolor: 'grey.200',
        px: 0.5,
        borderRadius: 0.5,
        fontFamily: 'monospace',
        fontSize: '0.85em',
      }}
    >
      {children}
    </Box>
  ),
  pre: ({ children }: { children?: React.ReactNode }) => (
    <Box
      component="pre"
      sx={{
        bgcolor: 'grey.900',
        color: 'grey.100',
        p: 1,
        borderRadius: 1,
        overflowX: 'auto',
        fontFamily: 'monospace',
        fontSize: '0.85em',
      }}
    >
      {children}
    </Box>
  ),
  blockquote: ({ children }: { children?: React.ReactNode }) => (
    <Box
      component="blockquote"
      sx={{ borderLeft: '3px solid', borderColor: 'grey.400', pl: 1.5, my: 1, color: 'text.secondary' }}
    >
      {children}
    </Box>
  ),
  a: ({ href, children }: { href?: string; children?: React.ReactNode }) => (
    <a href={href} target="_blank" rel="noopener noreferrer">
      {children}
    </a>
  ),
};

const MessageList: React.FC = () => {
  const { messages } = useCopilotChatInternal();
  const { state, running } = useCoAgent<AgentState>({
    name: RUN_AGENT_NAME,
    initialState: { retrieved_paragraphs: [] },
  });

  // toolCallId -> truncated result text (for the "done" tooltip)
  const toolResults = useMemo(() => {
    const m = new Map<string, string>();
    for (const msg of messages) {
      if ((msg as any).role === 'tool') {
        const id = (msg as any).toolCallId;
        const content = (msg as any).content;
        if (id && typeof content === 'string') {
          m.set(id, truncate(content, MAX_RESULT_PREVIEW_LEN));
        }
      }
    }
    return m;
  }, [messages]);

  // Visible roles: user + assistant. Tool messages are folded into their
  // matching tool-call cards via toolCallId.
  const visible = messages.filter(
    (m) => (m as any).role === 'user' || (m as any).role === 'assistant'
  );

  // Find last assistant index — that's where live citation chips attach.
  const lastAssistantIdx = (() => {
    for (let i = visible.length - 1; i >= 0; i--) {
      if ((visible[i] as any).role === 'assistant') return i;
    }
    return -1;
  })();

  const liveCitations = state?.retrieved_paragraphs ?? [];

  if (visible.length === 0 && !running) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        height="100%"
        color="text.secondary"
      >
        <Typography variant="h6">
          Start a conversation by typing a message below
        </Typography>
      </Box>
    );
  }

  return (
    <List sx={{ width: '100%', p: 1 }}>
      {visible.map((message, idx) => {
        const m = message as any;
        const isUser = m.role === 'user';
        const content =
          typeof m.content === 'string'
            ? m.content
            : Array.isArray(m.content)
            ? m.content
                .filter((c: any) => c?.type === 'text' && typeof c?.text === 'string')
                .map((c: any) => c.text)
                .join('')
            : '';
        const toolCalls = !isUser && Array.isArray(m.toolCalls) ? m.toolCalls : [];
        const showCitations = idx === lastAssistantIdx && liveCitations.length > 0;

        return (
          <ListItem
            key={m.id ?? `msg-${idx}`}
            alignItems="flex-start"
            sx={{ mb: 2, justifyContent: isUser ? 'flex-end' : 'flex-start' }}
          >
            <Box
              sx={{
                display: 'flex',
                flexDirection: isUser ? 'row-reverse' : 'row',
                alignItems: 'flex-start',
                maxWidth: '80%',
                width: 'fit-content',
              }}
            >
              <Avatar sx={{ bgcolor: isUser ? 'primary.main' : 'secondary.main', mx: 1 }}>
                {isUser ? <PersonIcon /> : <BotIcon />}
              </Avatar>

              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5, minWidth: 0 }}>
                {/* Tool-call cards (live-only, for assistant turns) */}
                {toolCalls.length > 0 && (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {toolCalls.map((tc: any) => {
                      const id = tc?.id;
                      const fn = tc?.function ?? {};
                      const name = fn.name ?? 'tool';
                      const rawArgs = fn.arguments;
                      const args = typeof rawArgs === 'string' && rawArgs.length > 0
                        ? (() => {
                            try { return JSON.parse(rawArgs); } catch { return rawArgs; }
                          })()
                        : rawArgs;
                      const result = id ? toolResults.get(id) : undefined;
                      return (
                        <ToolCallCard
                          key={id ?? `${name}-${rawArgs}`}
                          toolName={name}
                          argsPreview={argsPreview(args)}
                          done={!!result}
                          resultPreview={result}
                        />
                      );
                    })}
                  </Box>
                )}

                {/* Message bubble */}
                {content && (
                  <Paper
                    elevation={1}
                    sx={{
                      p: 2,
                      bgcolor: isUser ? 'primary.light' : 'grey.100',
                      color: isUser ? 'primary.contrastText' : 'text.primary',
                      borderRadius: 2,
                      wordBreak: 'break-word',
                    }}
                  >
                    {isUser ? (
                      <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
                        {content}
                      </Typography>
                    ) : (
                      <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                        {content}
                      </ReactMarkdown>
                    )}

                    {/* Citation chips — live-only, attached to last assistant message */}
                    {!isUser && showCitations && (
                      <Box sx={{ mt: 1 }}>
                        <Typography
                          variant="caption"
                          color="text.secondary"
                          sx={{ mb: 0.5, display: 'block' }}
                        >
                          Sources:
                        </Typography>
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                          {liveCitations.map((p, i) => (
                            <Chip
                              key={`${p.book_index}-${p.chapter_index}-${p.page}-${i}`}
                              label={citationLabel(p)}
                              size="small"
                              variant="outlined"
                              sx={{ fontSize: '0.75rem', height: 'auto' }}
                            />
                          ))}
                        </Box>
                      </Box>
                    )}
                  </Paper>
                )}
              </Box>
            </Box>
          </ListItem>
        );
      })}
    </List>
  );
};

export default MessageList;
