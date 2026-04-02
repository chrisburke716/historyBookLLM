import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { SearchResult, ENTITY_TYPE_COLORS } from '../types/kg';

interface GraphState {
  focusEntityId: string | null;
  graphName: string;
  displayMode: 'full' | 'nhop';
  hopCount: 1 | 2 | 3;
  searchResults: SearchResult[];
  colorMap: Record<string, string>;
  // History (stubbed — buttons rendered but disabled)
  focusHistory: string[];
  historyIndex: number;
}

const initialState: GraphState = {
  focusEntityId: null,
  graphName: 'volume_full',
  displayMode: 'full',
  hopCount: 2,
  searchResults: [],
  colorMap: ENTITY_TYPE_COLORS,
  focusHistory: [],
  historyIndex: -1,
};

const graphSlice = createSlice({
  name: 'graph',
  initialState,
  reducers: {
    setFocus(state, action: PayloadAction<string>) {
      state.focusEntityId = action.payload;
      // Append to history, truncating any forward entries
      const next = state.focusHistory.slice(0, state.historyIndex + 1);
      next.push(action.payload);
      state.focusHistory = next;
      state.historyIndex = next.length - 1;
    },
    clearFocus(state) {
      state.focusEntityId = null;
      state.displayMode = 'full';
    },
    setGraphName(state, action: PayloadAction<string>) {
      state.graphName = action.payload;
      state.focusEntityId = null;
      state.displayMode = 'full';
    },
    setDisplayMode(state, action: PayloadAction<'full' | 'nhop'>) {
      state.displayMode = action.payload;
    },
    setHopCount(state, action: PayloadAction<1 | 2 | 3>) {
      state.hopCount = action.payload;
    },
    setSearchResults(state, action: PayloadAction<SearchResult[]>) {
      state.searchResults = action.payload;
    },
    setColorMap(state, action: PayloadAction<Record<string, string>>) {
      state.colorMap = action.payload;
    },
  },
});

export const {
  setFocus,
  clearFocus,
  setGraphName,
  setDisplayMode,
  setHopCount,
  setSearchResults,
  setColorMap,
} = graphSlice.actions;

export default graphSlice.reducer;
