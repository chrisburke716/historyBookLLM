/**
 * BookPage - Main page for browsing and reading book content.
 */

import React, { useEffect, useState, useRef, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Container, Box, Alert, Snackbar } from '@mui/material';
import BookSelector from '../components/BookSelector';
import ChapterView from '../components/ChapterView';
import { chatAPI } from '../services/api';
import { ChapterContentResponse } from '../types';

// Maximum number of scroll positions to keep in localStorage
const MAX_SAVED_POSITIONS = 10;

// Debounce utility to avoid excessive localStorage writes
function debounce<T extends (...args: any[]) => void>(func: T, wait: number): T {
  let timeout: NodeJS.Timeout;
  return ((...args: any[]) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  }) as T;
}

// Clean up old scroll positions from localStorage
function cleanupOldScrollPositions(): void {
  try {
    // Get all scroll position keys with their timestamps
    const scrollKeys: { key: string; timestamp: number }[] = [];

    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key?.startsWith('book-scroll-')) {
        const value = localStorage.getItem(key);
        if (value) {
          try {
            const data = JSON.parse(value);
            scrollKeys.push({ key, timestamp: data.timestamp || 0 });
          } catch {
            // Old format (just position number) - assume timestamp 0
            scrollKeys.push({ key, timestamp: 0 });
          }
        }
      }
    }

    // Sort by timestamp (oldest first) and remove excess
    if (scrollKeys.length > MAX_SAVED_POSITIONS) {
      scrollKeys
        .sort((a, b) => a.timestamp - b.timestamp)
        .slice(0, scrollKeys.length - MAX_SAVED_POSITIONS)
        .forEach(({ key }) => localStorage.removeItem(key));
    }
  } catch (error) {
    console.error('Failed to cleanup scroll positions:', error);
  }
}

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

  // Ref for scroll container
  const scrollContainerRef = useRef<HTMLDivElement>(null);

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

  // Save scroll position to localStorage (debounced)
  const saveScrollPosition = useCallback(
    debounce(() => {
      if (selectedBookIndex === null || selectedChapterIndex === null) return;
      if (!scrollContainerRef.current) return;

      const position = scrollContainerRef.current.scrollTop;
      const key = `book-scroll-${selectedBookIndex}-${selectedChapterIndex}`;

      // Store position with timestamp for cleanup
      const data = {
        position,
        timestamp: Date.now(),
      };
      localStorage.setItem(key, JSON.stringify(data));

      // Clean up old positions
      cleanupOldScrollPositions();
    }, 300),
    [selectedBookIndex, selectedChapterIndex]
  );

  // Restore scroll position from localStorage when chapter loads
  useEffect(() => {
    if (!chapterContent || !scrollContainerRef.current) return;
    if (selectedBookIndex === null || selectedChapterIndex === null) return;

    const key = `book-scroll-${selectedBookIndex}-${selectedChapterIndex}`;
    const savedData = localStorage.getItem(key);

    if (savedData) {
      // Delay to ensure content is fully rendered
      setTimeout(() => {
        try {
          const { position } = JSON.parse(savedData);
          scrollContainerRef.current?.scrollTo({
            top: position,
            behavior: 'smooth',
          });
        } catch {
          // Invalid format - scroll to top
          scrollContainerRef.current?.scrollTo({ top: 0 });
        }
      }, 100);
    } else {
      // New chapter - scroll to top
      scrollContainerRef.current.scrollTo({ top: 0 });
    }
  }, [chapterContent, selectedBookIndex, selectedChapterIndex]);

  // Attach scroll event listener
  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container) return;

    const handleScroll = () => saveScrollPosition();
    container.addEventListener('scroll', handleScroll);

    return () => container.removeEventListener('scroll', handleScroll);
  }, [saveScrollPosition]);

  return (
    <Container
      maxWidth="lg"
      sx={{
        py: 4,
        height: 'calc(100vh - 64px)', // Full height minus AppBar (64px)
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden', // Prevent outer scroll
      }}
    >
      {/* Book and Chapter Selectors - Fixed at top */}
      <BookSelector
        selectedBookIndex={selectedBookIndex}
        selectedChapterIndex={selectedChapterIndex}
        onSelectionChange={handleSelectionChange}
        disabled={isLoadingContent}
      />

      {/* Scrollable Chapter Content Container */}
      <Box
        ref={scrollContainerRef}
        sx={{
          flex: 1,
          overflow: 'auto', // Enable scrolling within this container
          mt: 2,
        }}
      >
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
    </Container>
  );
};

export default BookPage;
