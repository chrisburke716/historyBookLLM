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
  ListSubheader,
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
import {
  setNodeSizeMetric,
  setNodeColorMetric,
  setNodeSizeParams,
  setNodeColorParams,
} from '../../store/metricsSlice';
import {
  useGraphListQuery,
  useBooksWithChaptersQuery,
  useNodeMetricQuery,
  useNodePairMetricQuery,
} from '../../hooks/useKGQueries';
import { kgAPI } from '../../services/kgAPI';
import {
  KGGraphMeta,
  ENTITY_TYPE_COLORS,
  NODE_PAIR_METRICS,
  NodeSizeMetric,
  NodeColorMetric,
  NodePairMetric,
} from '../../types/kg';

function buildGraphLabel(
  graph: KGGraphMeta,
  bookTitles: Map<number, string>,
  chapterTitles: Map<string, string>
): string {
  if (graph.graph_type === 'volume') return 'Full Volume';
  if (graph.graph_type === 'book') {
    const bookNum = parseInt(graph.name.replace('book', ''), 10);
    const title = bookTitles.get(bookNum);
    return title ? `Book ${bookNum}: ${title}` : `Book ${bookNum}`;
  }
  // chapter
  if (graph.book_chapters.length > 0) {
    const [bookStr, chapterStr] = graph.book_chapters[0].split(':');
    const [bookNum, chapterNum] = [parseInt(bookStr, 10), parseInt(chapterStr, 10)];
    const title = chapterTitles.get(`${bookNum}:${chapterNum}`);
    return title ? `Book ${bookNum} · Ch. ${chapterNum}: ${title}` : `Book ${bookNum} · Ch. ${chapterNum}`;
  }
  return graph.name;
}

const KGTopBar: React.FC = () => {
  const dispatch = useAppDispatch();
  const { graphName, displayMode, hopCount, searchResults, focusHistory, historyIndex, occurrenceThreshold, trimLeaves, focusEntityId } =
    useAppSelector((s) => s.graph);
  const { nodeSizeMetric, nodeColorMetric, nodeSizeParams, nodeColorParams } =
    useAppSelector((s) => s.metrics);

  const { data: graphList, isLoading: graphsLoading } = useGraphListQuery();
  const { bookTitles, chapterTitles } = useBooksWithChaptersQuery();

  // Size metric query (for loading indicator)
  const isNodePairColorMetric = NODE_PAIR_METRICS.includes(nodeColorMetric as NodePairMetric);
  const sizeMetricQuery = useNodeMetricQuery(
    graphName, nodeSizeMetric, nodeSizeParams,
    nodeSizeMetric !== NodeSizeMetric.OccurrenceCount
  );
  const colorMetricQuery = useNodeMetricQuery(
    graphName, nodeColorMetric as NodeColorMetric, nodeColorParams,
    !isNodePairColorMetric && nodeColorMetric !== NodeColorMetric.EntityType
  );
  const colorPairMetricQuery = useNodePairMetricQuery(
    graphName, focusEntityId, nodeColorMetric as NodePairMetric,
    isNodePairColorMetric && focusEntityId !== null
  );
  const sizeLoading = sizeMetricQuery.isFetching || sizeMetricQuery.data?.status === 'computing';
  const colorLoading = colorMetricQuery.isFetching || colorMetricQuery.data?.status === 'computing'
    || colorPairMetricQuery.isFetching;

  // Param input local state (committed on enter/blur)
  const [dampingInput, setDampingInput] = useState(String(nodeSizeParams.damping ?? 0.85));
  const [kInput, setKInput] = useState(String(nodeColorParams.k ?? 5));

  const [searchInput, setSearchInput] = useState('');
  const [searchOpen, setSearchOpen] = useState(false);
  const [searching, setSearching] = useState(false);

  const commitDamping = useCallback(() => {
    const val = parseFloat(dampingInput);
    if (!isNaN(val) && val > 0 && val < 1) {
      dispatch(setNodeSizeParams({ ...nodeSizeParams, damping: val }));
    }
  }, [dampingInput, nodeSizeParams, dispatch]);

  const commitK = useCallback(() => {
    const val = parseInt(kInput, 10);
    if (!isNaN(val) && val >= 2) {
      dispatch(setNodeColorParams({ ...nodeColorParams, k: val }));
    }
  }, [kInput, nodeColorParams, dispatch]);

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

  // Group and sort graphs for dropdown
  const volumeGraphs = graphList?.graphs.filter((g) => g.graph_type === 'volume') ?? [];
  const bookGraphs = (graphList?.graphs.filter(
    (g) => g.graph_type === 'book' && /^book\d+$/.test(g.name)
  ) ?? []).sort((a, b) => {
    const bookNum = (name: string) => parseInt(name.replace('book', ''), 10);
    return bookNum(a.name) - bookNum(b.name);
  });
  const chapterGraphs = (graphList?.graphs.filter((g) => g.graph_type === 'chapter') ?? [])
    .sort((a, b) => {
      const [ab, ac] = (a.book_chapters[0] ?? '0:0').split(':').map(Number);
      const [bb, bc] = (b.book_chapters[0] ?? '0:0').split(':').map(Number);
      return ab !== bb ? ab - bb : ac - bc;
    });

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
              {buildGraphLabel(g, bookTitles, chapterTitles)}
            </MenuItem>
          ))}
          {bookGraphs.length > 0 && (
            <MenuItem disabled sx={{ fontSize: '0.75rem', color: 'text.secondary', fontStyle: 'italic' }}>
              — Books —
            </MenuItem>
          )}
          {bookGraphs.map((g) => (
            <MenuItem key={g.name} value={g.name}>
              {buildGraphLabel(g, bookTitles, chapterTitles)}
            </MenuItem>
          ))}
          {chapterGraphs.length > 0 && (
            <MenuItem disabled sx={{ fontSize: '0.75rem', color: 'text.secondary', fontStyle: 'italic' }}>
              — Chapters —
            </MenuItem>
          )}
          {chapterGraphs.map((g) => (
            <MenuItem key={g.name} value={g.name}>
              {buildGraphLabel(g, bookTitles, chapterTitles)}
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      {/* Node size metric selector */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
        <FormControl size="small" sx={{ minWidth: 160 }}>
          <InputLabel>Size</InputLabel>
          <Select
            label="Size"
            value={nodeSizeMetric}
            onChange={(e) => dispatch(setNodeSizeMetric(e.target.value as NodeSizeMetric))}
            endAdornment={
              sizeLoading ? <CircularProgress size={14} sx={{ mr: 2 }} /> : null
            }
          >
            <MenuItem value={NodeSizeMetric.OccurrenceCount}>Occurrence count</MenuItem>
            <MenuItem value={NodeSizeMetric.DegreeCentrality}>Degree centrality</MenuItem>
            <MenuItem value={NodeSizeMetric.BetweennessCentrality}>Betweenness centrality</MenuItem>
            <MenuItem value={NodeSizeMetric.PageRank}>PageRank</MenuItem>
            <MenuItem value={NodeSizeMetric.ClosenessCentrality}>Closeness centrality</MenuItem>
            <MenuItem value={NodeSizeMetric.KCoreNumber}>K-core number</MenuItem>
          </Select>
        </FormControl>
        {nodeSizeMetric === NodeSizeMetric.PageRank && (
          <Tooltip title="Damping factor (0–1)">
            <TextField
              size="small"
              label="Damping"
              type="number"
              value={dampingInput}
              onChange={(e) => setDampingInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && commitDamping()}
              onBlur={commitDamping}
              inputProps={{ min: 0.01, max: 0.99, step: 0.05 }}
              sx={{ width: 90 }}
            />
          </Tooltip>
        )}
      </Box>

      {/* Node color metric selector */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
        <FormControl size="small" sx={{ minWidth: 180 }}>
          <InputLabel>Color</InputLabel>
          <Select
            label="Color"
            value={nodeColorMetric}
            onChange={(e) => dispatch(setNodeColorMetric(e.target.value as NodeColorMetric | NodePairMetric))}
            endAdornment={
              colorLoading ? <CircularProgress size={14} sx={{ mr: 2 }} /> : null
            }
          >
            <MenuItem value={NodeColorMetric.EntityType}>Entity type</MenuItem>
            <MenuItem value={NodeColorMetric.CommunityLouvain}>Community — Louvain</MenuItem>
            <MenuItem value={NodeColorMetric.CommunityGirvanNewman}>Community — Girvan-Newman</MenuItem>
            <MenuItem value={NodeColorMetric.CommunityLabelPropagation}>Community — Label prop.</MenuItem>
            <MenuItem value={NodeColorMetric.CommunitySpectral}>Community — Spectral</MenuItem>
            <MenuItem value={NodeColorMetric.LocalClusteringCoefficient}>Local clustering coeff.</MenuItem>
            <MenuItem value={NodeColorMetric.KCoreNumber}>K-core number</MenuItem>
            <ListSubheader sx={{ fontSize: '0.72rem', lineHeight: '1.6' }}>
              Focus-relative (select a node first)
            </ListSubheader>
            {(NODE_PAIR_METRICS as string[]).map((m) => (
              <MenuItem
                key={m}
                value={m}
                disabled={focusEntityId === null}
              >
                {m.replace(/_/g, ' ')}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
        {nodeColorMetric === NodeColorMetric.CommunitySpectral && (
          <Tooltip title="Number of clusters (k ≥ 2)">
            <TextField
              size="small"
              label="k"
              type="number"
              value={kInput}
              onChange={(e) => setKInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && commitK()}
              onBlur={commitK}
              inputProps={{ min: 2, max: 20, step: 1 }}
              sx={{ width: 70 }}
            />
          </Tooltip>
        )}
      </Box>

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
