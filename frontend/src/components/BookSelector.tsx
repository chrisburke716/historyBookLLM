/**
 * BookSelector - Cascading dropdown for selecting book and chapter.
 */

import React, { useEffect, useState } from 'react';
import {
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  CircularProgress,
  Alert,
  SelectChangeEvent,
} from '@mui/material';
import { chatAPI } from '../services/api';
import { BookResponse, ChapterResponse } from '../types';

interface BookSelectorProps {
  selectedBookIndex: number | null;
  selectedChapterIndex: number | null;
  onSelectionChange: (bookIndex: number | null, chapterIndex: number | null) => void;
  disabled?: boolean;
}

const BookSelector: React.FC<BookSelectorProps> = ({
  selectedBookIndex,
  selectedChapterIndex,
  onSelectionChange,
  disabled = false,
}) => {
  const [books, setBooks] = useState<BookResponse[]>([]);
  const [chapters, setChapters] = useState<ChapterResponse[]>([]);
  const [isLoadingBooks, setIsLoadingBooks] = useState(false);
  const [isLoadingChapters, setIsLoadingChapters] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load books on mount
  useEffect(() => {
    const loadBooks = async () => {
      setIsLoadingBooks(true);
      setError(null);
      try {
        const response = await chatAPI.getBooks();
        setBooks(response.books);
      } catch (err) {
        setError('Failed to load books');
        console.error('Error loading books:', err);
      } finally {
        setIsLoadingBooks(false);
      }
    };

    loadBooks();
  }, []);

  // Load chapters when book is selected
  useEffect(() => {
    if (selectedBookIndex === null) {
      setChapters([]);
      return;
    }

    const loadChapters = async () => {
      setIsLoadingChapters(true);
      setError(null);
      try {
        const response = await chatAPI.getChapters(selectedBookIndex);
        setChapters(response.chapters);
      } catch (err) {
        setError('Failed to load chapters');
        console.error('Error loading chapters:', err);
      } finally {
        setIsLoadingChapters(false);
      }
    };

    loadChapters();
  }, [selectedBookIndex]);

  const handleBookChange = (event: SelectChangeEvent<number>) => {
    const bookIndex = event.target.value as number;
    onSelectionChange(bookIndex, null); // Reset chapter when book changes
  };

  const handleChapterChange = (event: SelectChangeEvent<number>) => {
    const chapterIndex = event.target.value as number;
    onSelectionChange(selectedBookIndex, chapterIndex);
  };

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        {error}
      </Alert>
    );
  }

  return (
    <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
      {/* Book Selector */}
      <FormControl sx={{ minWidth: 300 }} disabled={disabled || isLoadingBooks}>
        <InputLabel id="book-select-label">Select Book</InputLabel>
        <Select
          labelId="book-select-label"
          id="book-select"
          value={selectedBookIndex ?? ''}
          label="Select Book"
          onChange={handleBookChange}
        >
          {isLoadingBooks ? (
            <MenuItem disabled>
              <CircularProgress size={20} sx={{ mr: 1 }} />
              Loading books...
            </MenuItem>
          ) : (
            books.map((book) => (
              <MenuItem key={book.id} value={book.book_index}>
                {book.title}
              </MenuItem>
            ))
          )}
        </Select>
      </FormControl>

      {/* Chapter Selector */}
      <FormControl
        sx={{ minWidth: 400 }}
        disabled={disabled || !selectedBookIndex || isLoadingChapters}
      >
        <InputLabel id="chapter-select-label">Select Chapter</InputLabel>
        <Select
          labelId="chapter-select-label"
          id="chapter-select"
          value={selectedChapterIndex ?? ''}
          label="Select Chapter"
          onChange={handleChapterChange}
        >
          {isLoadingChapters ? (
            <MenuItem disabled>
              <CircularProgress size={20} sx={{ mr: 1 }} />
              Loading chapters...
            </MenuItem>
          ) : selectedBookIndex === null ? (
            <MenuItem disabled>First select a book</MenuItem>
          ) : (
            chapters.map((chapter) => (
              <MenuItem key={chapter.id} value={chapter.chapter_index}>
                {chapter.title}
              </MenuItem>
            ))
          )}
        </Select>
      </FormControl>
    </Box>
  );
};

export default BookSelector;
