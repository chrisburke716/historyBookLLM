import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { NodeColorMetric, NodePairMetric, NodeSizeMetric } from '../types/kg';

interface MetricsState {
  nodeSizeMetric: NodeSizeMetric;
  nodeColorMetric: NodeColorMetric | NodePairMetric;
  nodeSizeParams: Record<string, number>; // e.g. { damping: 0.85 }
  nodeColorParams: Record<string, number>; // e.g. { k: 5 }
}

const initialState: MetricsState = {
  nodeSizeMetric: NodeSizeMetric.OccurrenceCount,
  nodeColorMetric: NodeColorMetric.EntityType,
  nodeSizeParams: { damping: 0.85 },
  nodeColorParams: { k: 5 },
};

const metricsSlice = createSlice({
  name: 'metrics',
  initialState,
  reducers: {
    setNodeSizeMetric(state, action: PayloadAction<NodeSizeMetric>) {
      state.nodeSizeMetric = action.payload;
    },
    setNodeColorMetric(state, action: PayloadAction<NodeColorMetric | NodePairMetric>) {
      state.nodeColorMetric = action.payload;
    },
    setNodeSizeParams(state, action: PayloadAction<Record<string, number>>) {
      state.nodeSizeParams = action.payload;
    },
    setNodeColorParams(state, action: PayloadAction<Record<string, number>>) {
      state.nodeColorParams = action.payload;
    },
  },
});

export const {
  setNodeSizeMetric,
  setNodeColorMetric,
  setNodeSizeParams,
  setNodeColorParams,
} = metricsSlice.actions;

export default metricsSlice.reducer;
