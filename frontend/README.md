# History Book Chat Frontend

A React TypeScript frontend providing an intuitive chat interface for conversational interactions with historical documents.

## Features

- **💬 Chat Interface**: Clean, modern chat UI with message history
- **🎨 Material-UI Design**: Responsive design with Material-UI components
- **📱 Session Management**: Create, switch between, and manage conversation sessions
- **⚡ Real-time Responses**: Live chat with loading states and error handling
- **🔍 Citations**: Display source references from historical documents
- **♿ Accessibility**: Built with accessibility best practices

## Tech Stack

- **React 19** with TypeScript
- **Material-UI (MUI)** for components and theming
- **Axios** for API communication
- **Create React App** for build tooling

## Quick Start

### Prerequisites

- Node.js 16+
- npm or yarn
- Backend API running on port 8000

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm start
```

The application will open at [http://localhost:3000](http://localhost:3000).

### Development

```bash
# Run tests
npm test

# Build for production
npm run build

# Type checking
npx tsc --noEmit
```

## Architecture

### Components

- **MessageInput.tsx**: Input field and send functionality
- **MessageList.tsx**: Chat message display with citations
- **SessionDropdown.tsx**: Session selection and management
- **App.tsx**: Main application layout and state management

### Services

- **api.ts**: HTTP client for backend communication
- Handles authentication, error responses, and data formatting

### State Management

- React hooks for local component state
- Session and message data managed through API calls
- Real-time updates through REST polling

## API Integration

The frontend communicates with the FastAPI backend:

- **POST /chat/sessions**: Create new chat sessions
- **GET /chat/sessions**: List user sessions
- **POST /chat/sessions/{id}/messages**: Send messages
- **GET /chat/sessions/{id}/messages**: Retrieve message history

## Environment Configuration

The frontend expects the backend to be available at:
- Development: `http://localhost:8000`
- Production: Configure via build process

## Build & Deployment

```bash
# Production build
npm run build

# Serve static files
npx serve -s build -l 3000
```

Build outputs to `build/` directory, ready for static hosting.

## Development Workflow

1. Ensure backend API is running on port 8000
2. Start frontend development server: `npm start`
3. Changes auto-reload in browser
4. Use browser dev tools for debugging
5. Run tests before committing: `npm test`