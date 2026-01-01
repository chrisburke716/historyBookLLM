/**
 * ChapterView - Display chapter content with paragraphs and page numbers.
 */

import React from 'react';
import { Box, Typography, Paper, CircularProgress } from '@mui/material';
import { ChapterContentResponse } from '../types';

interface ChapterViewProps {
  chapterContent: ChapterContentResponse | null;
  isLoading: boolean;
}

const ChapterView: React.FC<ChapterViewProps> = ({ chapterContent, isLoading }) => {
  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (!chapterContent) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        minHeight="400px"
        color="text.secondary"
      >
        <Typography variant="body1">
          Select a book and chapter to start reading
        </Typography>
      </Box>
    );
  }

  const { chapter, paragraphs } = chapterContent;

  return (
    <Paper elevation={1} sx={{ p: 4, maxWidth: 900, margin: '0 auto' }}>
      {/* Chapter Title */}
      <Typography
        variant="h4"
        component="h1"
        gutterBottom
        sx={{ mb: 4, fontWeight: 600, borderBottom: '2px solid #1976d2', pb: 2 }}
      >
        {chapter.title}
      </Typography>

      {/* Chapter Metadata */}
      <Typography
        variant="body2"
        color="text.secondary"
        sx={{ mb: 4, fontStyle: 'italic' }}
      >
        Pages {chapter.start_page}–{chapter.end_page}
      </Typography>

      {/* Paragraphs with Page Numbers */}
      <Box>
        {paragraphs.length === 0 ? (
          <Typography color="text.secondary">
            No content available for this chapter.
          </Typography>
        ) : (
          paragraphs.map((paragraph, index) => (
            <Box
              key={index}
              sx={{
                display: 'flex',
                mb: 3,
                alignItems: 'flex-start',
              }}
            >
              {/* Page Number in Left Margin */}
              <Box
                sx={{
                  minWidth: 60,
                  textAlign: 'right',
                  pr: 2,
                  color: 'text.secondary',
                  fontSize: '0.875rem',
                  fontWeight: 500,
                  pt: 0.5,
                }}
              >
                {paragraph.page}
              </Box>

              {/* Paragraph Text */}
              <Typography
                variant="body1"
                component="p"
                sx={{
                  flex: 1,
                  lineHeight: 1.8,
                  textAlign: 'justify',
                }}
              >
                {paragraph.text}
              </Typography>
            </Box>
          ))
        )}
      </Box>
    </Paper>
  );
};

export default ChapterView;
