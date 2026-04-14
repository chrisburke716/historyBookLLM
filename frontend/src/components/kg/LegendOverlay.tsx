import React from 'react';
import { Box, Typography } from '@mui/material';
import { interpolateViridis } from 'd3-scale-chromatic';
import { ENTITY_TYPE_COLORS, ENTITY_TYPES } from '../../types/kg';

// Pre-compute viridis CSS gradient (10 stops, left=t0, right=t1)
const VIRIDIS_GRADIENT = (() => {
  const stops = Array.from({ length: 10 }, (_, i) => {
    const t = i / 9;
    return `${interpolateViridis(t)} ${(t * 100).toFixed(0)}%`;
  });
  return `linear-gradient(to right, ${stops.join(', ')})`;
})();

function formatVal(v: number): string {
  if (v === 0) return '0';
  const abs = Math.abs(v);
  if (abs >= 100) return v.toFixed(0);
  if (abs >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

interface LegendOverlayProps {
  /** null = community metric, no legend */
  mode: 'entity_type' | 'continuous' | null;
  // Continuous mode only
  colorNormMin?: number;
  colorNormMax?: number;
  label?: string;
}

const LegendOverlay: React.FC<LegendOverlayProps> = ({
  mode,
  colorNormMin = 0,
  colorNormMax = 1,
  label,
}) => {
  if (mode === null) return null;

  return (
    <Box
      sx={{
        position: 'absolute',
        bottom: 16,
        left: 16,
        zIndex: 10,
        bgcolor: 'rgba(255,255,255,0.88)',
        borderRadius: 1,
        px: 1.5,
        py: 1,
        boxShadow: 1,
        minWidth: 140,
        maxWidth: 200,
        pointerEvents: 'none',
      }}
    >
      {mode === 'entity_type' && (
        <>
          {ENTITY_TYPES.map((type) => (
            <Box key={type} sx={{ display: 'flex', alignItems: 'center', gap: 0.75, mb: 0.25 }}>
              <Box
                sx={{
                  width: 11,
                  height: 11,
                  borderRadius: '2px',
                  bgcolor: ENTITY_TYPE_COLORS[type],
                  flexShrink: 0,
                }}
              />
              <Typography variant="caption" sx={{ textTransform: 'capitalize' }}>
                {type}
              </Typography>
            </Box>
          ))}
        </>
      )}

      {mode === 'continuous' && (
        <>
          {label && (
            <Typography variant="caption" sx={{ display: 'block', mb: 0.75, fontWeight: 500, lineHeight: 1.2 }}>
              {label}
            </Typography>
          )}
          <Box
            sx={{
              height: 10,
              borderRadius: '2px',
              background: VIRIDIS_GRADIENT,
              mb: 0.25,
            }}
          />
          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
              {formatVal(colorNormMin)}
            </Typography>
            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
              {formatVal(colorNormMax)}
            </Typography>
          </Box>
        </>
      )}
    </Box>
  );
};

export default LegendOverlay;
