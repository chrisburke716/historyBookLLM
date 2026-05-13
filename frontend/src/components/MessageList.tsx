/**
 * MessageList — renders the conversation as a sequence of turns.
 *
 * A turn is one user question + the agent's reaction: any number of tool
 * calls + (optionally) one final text answer. The AG-UI stream can split
 * that across multiple assistant messages mid-run and consolidate them
 * post-run; rendering by turn instead of by message gives the same visual
 * structure regardless of how the underlying messages array is grouped.
 *
 * Per-turn layout:
 *   - User bubble (right, user avatar)
 *   - Tool-call chips (left, no avatar, compact stack)
 *   - Either:
 *       * Assistant bubble (left, bot avatar, markdown + citations on latest turn)
 *       * "Thinking…" indicator (latest turn only, while running w/ no text yet)
 */

import React from 'react';
import {
  Avatar,
  Box,
  Chip,
  CircularProgress,
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

interface ToolInvocation {
  id: string;
  name: string;
  args: unknown;
  result?: string;
}

interface Turn {
  userId: string;
  userText: string;
  toolCalls: ToolInvocation[];
  assistantId: string | null;
  assistantText: string;
}

const RUN_AGENT_NAME = 'rag';
const MAX_ARG_PREVIEW_LEN = 60;
const MAX_RESULT_PREVIEW_LEN = 240;

function truncate(s: string, n: number): string {
  return s.length > n ? s.slice(0, n - 1) + '…' : s;
}

function argsPreview(args: unknown): string | undefined {
  if (!args) return undefined;
  if (typeof args === 'string') return truncate(args, MAX_ARG_PREVIEW_LEN);
  try {
    return truncate(JSON.stringify(args), MAX_ARG_PREVIEW_LEN);
  } catch {
    return undefined;
  }
}

function citationLabel(p: ParagraphState): string {
  return `[B${p.book_index}, Ch${p.chapter_index}, p.${p.page}]`;
}

function extractText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content
      .filter((c: any) => c?.type === 'text' && typeof c?.text === 'string')
      .map((c: any) => c.text)
      .join('');
  }
  return '';
}

/**
 * Group a flat AG-UI message array into turns.
 *
 * UserMessage → starts a new turn.
 * AssistantMessage → appends text to current turn's assistantText and any
 *   `toolCalls` to current turn's toolCalls list.
 * ToolMessage → matched by toolCallId; supplies the result preview for the
 *   matching ToolInvocation in the current turn.
 *
 * Tolerant of the in-progress shape (assistant messages without text, tool
 * calls without results yet) — those just produce partial turns.
 */
function groupIntoTurns(messages: any[]): Turn[] {
  const turns: Turn[] = [];
  let current: Turn | null = null;

  for (const msg of messages) {
    const role = msg?.role;
    if (role === 'user') {
      current = {
        userId: msg.id ?? `user-${turns.length}`,
        userText: extractText(msg.content),
        toolCalls: [],
        assistantId: null,
        assistantText: '',
      };
      turns.push(current);
    } else if (role === 'assistant' && current) {
      const text = extractText(msg.content);
      if (text) {
        current.assistantText += text;
        current.assistantId = msg.id ?? current.assistantId;
      }
      const toolCalls = Array.isArray(msg.toolCalls) ? msg.toolCalls : [];
      for (const tc of toolCalls) {
        const id = tc?.id;
        if (!id) continue;
        const fn = tc?.function ?? {};
        const name = fn.name ?? 'tool';
        const rawArgs = fn.arguments;
        const args =
          typeof rawArgs === 'string' && rawArgs.length > 0
            ? (() => {
                try {
                  return JSON.parse(rawArgs);
                } catch {
                  return rawArgs;
                }
              })()
            : rawArgs;
        current.toolCalls.push({ id, name, args });
      }
    } else if (role === 'tool' && current) {
      const id = msg.toolCallId;
      const content = typeof msg.content === 'string' ? msg.content : '';
      const tc = current.toolCalls.find((t) => t.id === id);
      if (tc) tc.result = truncate(content, MAX_RESULT_PREVIEW_LEN);
    }
  }
  return turns;
}

const markdownComponents = {
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
    <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>
  ),
};

// Left padding for tool-call rows. Aligns the chip stack with where a bot
// bubble would start (past the avatar gutter).
const BOT_GUTTER_PL = 8;

const MessageList: React.FC = () => {
  const { messages } = useCopilotChatInternal();
  const { state, running } = useCoAgent<AgentState>({
    name: RUN_AGENT_NAME,
    initialState: { retrieved_paragraphs: [] },
  });

  const turns = groupIntoTurns(messages as any[]);
  const liveCitations = state?.retrieved_paragraphs ?? [];

  if (turns.length === 0 && !running) {
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
      {turns.map((turn, idx) => {
        const isLatest = idx === turns.length - 1;
        const showThinking = isLatest && running && !turn.assistantText;
        const showCitations =
          isLatest && !!turn.assistantText && liveCitations.length > 0;

        return (
          <Box key={turn.userId} sx={{ mb: 3 }}>
            {/* User bubble — right, user avatar */}
            <ListItem
              alignItems="flex-start"
              sx={{ mb: 1, justifyContent: 'flex-end' }}
            >
              <Box
                sx={{
                  display: 'flex',
                  flexDirection: 'row-reverse',
                  alignItems: 'flex-start',
                  maxWidth: '80%',
                }}
              >
                <Avatar sx={{ bgcolor: 'primary.main', mx: 1 }}>
                  <PersonIcon />
                </Avatar>
                <Paper
                  elevation={1}
                  sx={{
                    p: 2,
                    bgcolor: 'primary.light',
                    color: 'primary.contrastText',
                    borderRadius: 2,
                    wordBreak: 'break-word',
                  }}
                >
                  <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
                    {turn.userText}
                  </Typography>
                </Paper>
              </Box>
            </ListItem>

            {/* Tool-call chips — left, no avatar, stacked */}
            {turn.toolCalls.length > 0 && (
              <ListItem alignItems="flex-start" sx={{ mb: 1, pl: BOT_GUTTER_PL }}>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                  {turn.toolCalls.map((tc) => (
                    <ToolCallCard
                      key={tc.id}
                      toolName={tc.name}
                      argsPreview={argsPreview(tc.args)}
                      resultPreview={tc.result}
                    />
                  ))}
                </Box>
              </ListItem>
            )}

            {/* Assistant bubble OR thinking indicator (mutually exclusive) */}
            {showThinking ? (
              <ListItem alignItems="flex-start" sx={{ mb: 1 }}>
                <Box sx={{ display: 'flex', flexDirection: 'row', alignItems: 'center' }}>
                  <Avatar sx={{ bgcolor: 'secondary.main', mx: 1 }}>
                    <BotIcon />
                  </Avatar>
                  <Paper
                    elevation={1}
                    sx={{
                      p: 2,
                      bgcolor: 'grey.100',
                      borderRadius: 2,
                      display: 'flex',
                      alignItems: 'center',
                      gap: 1,
                    }}
                  >
                    <CircularProgress size={14} />
                    <Typography variant="body2" color="text.secondary">
                      Thinking…
                    </Typography>
                  </Paper>
                </Box>
              </ListItem>
            ) : turn.assistantText ? (
              <ListItem alignItems="flex-start" sx={{ mb: 1 }}>
                <Box
                  sx={{
                    display: 'flex',
                    flexDirection: 'row',
                    alignItems: 'flex-start',
                    maxWidth: '80%',
                  }}
                >
                  <Avatar sx={{ bgcolor: 'secondary.main', mx: 1 }}>
                    <BotIcon />
                  </Avatar>
                  <Paper
                    elevation={1}
                    sx={{
                      p: 2,
                      bgcolor: 'grey.100',
                      color: 'text.primary',
                      borderRadius: 2,
                      wordBreak: 'break-word',
                    }}
                  >
                    <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                      {turn.assistantText}
                    </ReactMarkdown>

                    {showCitations && (
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
                </Box>
              </ListItem>
            ) : null}
          </Box>
        );
      })}
    </List>
  );
};

export default MessageList;
