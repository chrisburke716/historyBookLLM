import React, { useState, useCallback } from 'react';
import {
  Box,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  ToggleButton,
  ToggleButtonGroup,
  IconButton,
  Paper,
  List,
  ListItemButton,
  ListItemText,
  Chip,
  CircularProgress,
  Typography,
  Tooltip,
  Slider,
} from '@mui/material';
import {
  ArrowBack,
  ArrowForward,
  Search as SearchIcon,
} from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../../store';
import {
  setFocus,
  setGraphName,
  setDisplayMode,
  setHopCount,
  setSearchResults,
  setOccurrenceThreshold,
  setTrimLeaves,
} from '../../store/graphSlice';
import { useGraphListQuery } from '../../hooks/useKGQueries';
import { kgAPI } from '../../services/kgAPI';
import { KGGraphMeta, ENTITY_TYPE_COLORS } from '../../types/kg';

function buildGraphLabel(graph: KGGraphMeta): string {
  if (graph.graph_type === 'volume') return 'Full Volume';
  if (graph.graph_type === 'book') {
    const bookNums = Array.from(new Set(graph.book_chapters.map((bc) => bc.split(':')[0])));
    return `Book ${bookNums.join(', ')}`;
  }
  // chapter
  if (graph.book_chapters.length > 0) {
    const [book, chapter] = graph.book_chapters[0].split(':');
    return `Book ${book} · Ch. ${chapter}`;
  }
  return graph.name;
}

const KGTopBar: React.FC = () => {
  const dispatch = useAppDispatch();
  const { graphName, displayMode, hopCount, searchResults, focusHistory, historyIndex, occurrenceThreshold, trimLeaves } =
    useAppSelector((s) => s.graph);

  const { data: graphList, isLoading: graphsLoading } = useGraphListQuery();

  const [searchInput, setSearchInput] = useState('');
  const [searchOpen, setSearchOpen] = useState(false);
  const [searching, setSearching] = useState(false);

  const handleSearch = useCallback(
    async (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key !== 'Enter' || !searchInput.trim()) return;
      setSearching(true);
      try {
        const resp = await kgAPI.search({
          query: searchInput.trim(),
          graph_name: graphName || undefined,
          limit: 10,
        });
        dispatch(setSearchResults(resp.results));
        setSearchOpen(true);
      } finally {
        setSearching(false);
      }
    },
    [searchInput, graphName, dispatch]
  );

  const handleSelectResult = useCallback(
    (entityId: string) => {
      dispatch(setFocus(entityId));
      dispatch(setDisplayMode('nhop'));
      setSearchOpen(false);
      setSearchInput('');
      dispatch(setSearchResults([]));
    },
    [dispatch]
  );

  // Group graphs for dropdown
  const volumeGraphs = graphList?.graphs.filter((g) => g.graph_type === 'volume') ?? [];
  const bookGraphs = graphList?.graphs.filter((g) => g.graph_type === 'book') ?? [];
  const chapterGraphs = graphList?.graphs.filter((g) => g.graph_type === 'chapter') ?? [];

  const canGoBack = historyIndex > 0;
  const canGoForward = historyIndex < focusHistory.length - 1;

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 1.5,
        px: 2,
        py: 1,
        borderBottom: 1,
        borderColor: 'divider',
        flexWrap: 'wrap',
        position: 'relative',
      }}
    >
      {/* Search */}
      <Box sx={{ position: 'relative', width: 280 }}>
        <TextField
          size="small"
          placeholder="Search entities…"
          value={searchInput}
          onChange={(e) => setSearchInput(e.target.value)}
          onKeyDown={handleSearch}
          onFocus={() => searchResults.length > 0 && setSearchOpen(true)}
          onBlur={() => setTimeout(() => setSearchOpen(false), 150)}
          InputProps={{
            startAdornment: searching ? (
              <CircularProgress size={16} sx={{ mr: 1 }} />
            ) : (
              <SearchIcon fontSize="small" sx={{ mr: 0.5, color: 'text.secondary' }} />
            ),
          }}
          sx={{ width: '100%' }}
        />
        {searchOpen && searchResults.length > 0 && (
          <Paper
            elevation={4}
            sx={{ position: 'absolute', top: '100%', left: 0, right: 0, zIndex: 1300, maxHeight: 320, overflow: 'auto' }}
          >
            <List dense disablePadding>
              {searchResults.map((r) => (
                <ListItemButton key={r.id} onMouseDown={() => handleSelectResult(r.id)}>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <span>{r.name}</span>
                        <Chip
                          label={r.entity_type}
                          size="small"
                          sx={{
                            height: 18,
                            fontSize: '0.65rem',
                            bgcolor: ENTITY_TYPE_COLORS[r.entity_type] ?? '#999',
                            color: '#fff',
                          }}
                        />
                      </Box>
                    }
                    secondary={r.aliases.slice(0, 2).join(', ') || undefined}
                  />
                </ListItemButton>
              ))}
            </List>
          </Paper>
        )}
      </Box>

      {/* Scope selector */}
      <FormControl size="small" sx={{ minWidth: 200 }}>
        <InputLabel>Graph</InputLabel>
        <Select
          label="Graph"
          value={graphName}
          onChange={(e) => dispatch(setGraphName(e.target.value))}
          disabled={graphsLoading}
        >
          {volumeGraphs.map((g) => (
            <MenuItem key={g.name} value={g.name}>
              {buildGraphLabel(g)}
            </MenuItem>
          ))}
          {bookGraphs.length > 0 && (
            <MenuItem disabled sx={{ fontSize: '0.75rem', color: 'text.secondary', fontStyle: 'italic' }}>
              — Books —
            </MenuItem>
          )}
          {bookGraphs.map((g) => (
            <MenuItem key={g.name} value={g.name}>
              {buildGraphLabel(g)}
            </MenuItem>
          ))}
          {chapterGraphs.length > 0 && (
            <MenuItem disabled sx={{ fontSize: '0.75rem', color: 'text.secondary', fontStyle: 'italic' }}>
              — Chapters —
            </MenuItem>
          )}
          {chapterGraphs.map((g) => (
            <MenuItem key={g.name} value={g.name}>
              {buildGraphLabel(g)}
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      {/* N-hop toggle */}
      <ToggleButtonGroup
        size="small"
        exclusive
        value={displayMode}
        onChange={(_e, val) => val && dispatch(setDisplayMode(val))}
      >
        <ToggleButton value="full">Full</ToggleButton>
        <ToggleButton value="nhop">N-hop</ToggleButton>
      </ToggleButtonGroup>

      {/* Hop count */}
      <FormControl size="small" sx={{ minWidth: 80 }}>
        <InputLabel>Hops</InputLabel>
        <Select
          label="Hops"
          value={hopCount}
          onChange={(e) => dispatch(setHopCount(Number(e.target.value) as 1 | 2 | 3))}
          disabled={displayMode !== 'nhop'}
        >
          <MenuItem value={1}>1</MenuItem>
          <MenuItem value={2}>2</MenuItem>
          <MenuItem value={3}>3</MenuItem>
        </Select>
      </FormControl>

      {/* Occurrence threshold slider */}
      <Box sx={{ display: 'flex', flexDirection: 'column', width: 140, px: 1 }}>
        <Typography variant="caption" color="text.secondary" sx={{ lineHeight: 1 }}>
          Min. occurrences: {occurrenceThreshold === 1 ? 'all' : `≥ ${occurrenceThreshold}`}
        </Typography>
        <Slider
          size="small"
          min={1}
          max={4}
          step={1}
          value={occurrenceThreshold}
          onChange={(_e, val) => dispatch(setOccurrenceThreshold(val as number))}
          marks
          sx={{ py: 0.5 }}
        />
      </Box>

      {/* Leaf trim toggle */}
      <Tooltip title="Recursively remove nodes with only one connection">
        <ToggleButton
          value="trimLeaves"
          size="small"
          selected={trimLeaves}
          onChange={() => dispatch(setTrimLeaves(!trimLeaves))}
        >
          Trim leaves
        </ToggleButton>
      </Tooltip>

      {/* History navigation (stubbed) */}
      <Box sx={{ display: 'flex', gap: 0.5 }}>
        <Tooltip title="Back (coming soon)">
          <span>
            <IconButton size="small" disabled={!canGoBack}>
              <ArrowBack fontSize="small" />
            </IconButton>
          </span>
        </Tooltip>
        <Tooltip title="Forward (coming soon)">
          <span>
            <IconButton size="small" disabled={!canGoForward}>
              <ArrowForward fontSize="small" />
            </IconButton>
          </span>
        </Tooltip>
      </Box>

      {/* Graph info */}
      {graphsLoading && (
        <Typography variant="caption" color="text.secondary">
          Loading graphs…
        </Typography>
      )}
    </Box>
  );
};

export default KGTopBar;
