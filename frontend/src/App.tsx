import React from 'react';
import { BrowserRouter, Routes, Route, useLocation, useNavigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, AppBar, Toolbar, Typography, Box, Tabs, Tab } from '@mui/material';
import { Book as BookIcon, Chat as ChatIcon } from '@mui/icons-material';
import ChatPage from './pages/ChatPage';
import BookPage from './pages/BookPage';

// Create MUI theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    h6: {
      fontWeight: 600,
    },
  },
  shape: {
    borderRadius: 8,
  },
});

// Navigation component that uses routing hooks
function Navigation() {
  const location = useLocation();
  const navigate = useNavigate();

  const currentTab = location.pathname.startsWith('/book') ? '/book' : '/chat';

  const handleTabChange = (_event: React.SyntheticEvent, newValue: string) => {
    navigate(newValue);
  };

  return (
    <AppBar position="static" elevation={1}>
      <Toolbar>
        <BookIcon sx={{ mr: 2 }} />
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          History Book Chat
        </Typography>
        <Tabs
          value={currentTab}
          onChange={handleTabChange}
          textColor="inherit"
          indicatorColor="secondary"
          sx={{ ml: 'auto' }}
        >
          <Tab
            icon={<ChatIcon />}
            iconPosition="start"
            label="Chat"
            value="/chat"
            sx={{ minHeight: 64, color: 'inherit' }}
          />
          <Tab
            icon={<BookIcon />}
            iconPosition="start"
            label="Book"
            value="/book"
            sx={{ minHeight: 64, color: 'inherit' }}
          />
        </Tabs>
      </Toolbar>
    </AppBar>
  );
}

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <BrowserRouter>
        <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
          {/* Navigation Bar */}
          <Navigation />

          {/* Main Content - Routed Pages */}
          <Box component="main" sx={{ flex: 1 }}>
            <Routes>
              <Route path="/" element={<ChatPage />} />
              <Route path="/chat" element={<ChatPage />} />
              <Route path="/book" element={<BookPage />} />
              <Route path="/book/:bookIndex/:chapterIndex" element={<BookPage />} />
            </Routes>
          </Box>
        </Box>
      </BrowserRouter>
    </ThemeProvider>
  );
}

export default App;
