/**
 * BookPage - Main page for browsing and reading book content.
 */

import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Container, Box, Alert, Snackbar } from '@mui/material';
import BookSelector from '../components/BookSelector';
import ChapterView from '../components/ChapterView';
import { chatAPI } from '../services/api';
import { ChapterContentResponse } from '../types';

const BookPage: React.FC = () => {
  const { bookIndex: bookIndexParam, chapterIndex: chapterIndexParam } = useParams<{
    bookIndex?: string;
    chapterIndex?: string;
  }>();
  const navigate = useNavigate();

  const [selectedBookIndex, setSelectedBookIndex] = useState<number | null>(null);
  const [selectedChapterIndex, setSelectedChapterIndex] = useState<number | null>(null);
  const [chapterContent, setChapterContent] = useState<ChapterContentResponse | null>(null);
  const [isLoadingContent, setIsLoadingContent] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Initialize selection from URL params
  useEffect(() => {
    if (bookIndexParam && chapterIndexParam) {
      const bookIdx = parseInt(bookIndexParam, 10);
      const chapterIdx = parseInt(chapterIndexParam, 10);

      if (!isNaN(bookIdx) && !isNaN(chapterIdx)) {
        setSelectedBookIndex(bookIdx);
        setSelectedChapterIndex(chapterIdx);
      }
    }
  }, [bookIndexParam, chapterIndexParam]);

  // Load chapter content when both book and chapter are selected
  useEffect(() => {
    if (selectedBookIndex === null || selectedChapterIndex === null) {
      setChapterContent(null);
      return;
    }

    const loadChapterContent = async () => {
      setIsLoadingContent(true);
      setError(null);
      try {
        const content = await chatAPI.getChapterContent(selectedBookIndex, selectedChapterIndex);
        setChapterContent(content);
      } catch (err) {
        setError('Failed to load chapter content');
        console.error('Error loading chapter content:', err);
        setChapterContent(null);
      } finally {
        setIsLoadingContent(false);
      }
    };

    loadChapterContent();
  }, [selectedBookIndex, selectedChapterIndex]);

  // Handle selection changes and update URL
  const handleSelectionChange = (bookIndex: number | null, chapterIndex: number | null) => {
    setSelectedBookIndex(bookIndex);
    setSelectedChapterIndex(chapterIndex);

    // Update URL to persist state
    if (bookIndex !== null && chapterIndex !== null) {
      navigate(`/book/${bookIndex}/${chapterIndex}`, { replace: true });
    } else if (bookIndex !== null) {
      // Just book selected, stay on /book but update state
      navigate('/book', { replace: true });
    } else {
      // Nothing selected
      navigate('/book', { replace: true });
    }
  };

  const handleCloseError = () => {
    setError(null);
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '80vh' }}>
        {/* Book and Chapter Selectors */}
        <BookSelector
          selectedBookIndex={selectedBookIndex}
          selectedChapterIndex={selectedChapterIndex}
          onSelectionChange={handleSelectionChange}
          disabled={isLoadingContent}
        />

        {/* Chapter Content */}
        <Box sx={{ flex: 1 }}>
          <ChapterView chapterContent={chapterContent} isLoading={isLoadingContent} />
        </Box>

        {/* Error Snackbar */}
        <Snackbar
          open={!!error}
          autoHideDuration={6000}
          onClose={handleCloseError}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        >
          <Alert onClose={handleCloseError} severity="error" variant="filled" sx={{ width: '100%' }}>
            {error}
          </Alert>
        </Snackbar>
      </Box>
    </Container>
  );
};

export default BookPage;
