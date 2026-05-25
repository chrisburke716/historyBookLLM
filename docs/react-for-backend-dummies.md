# React for Backend Dummies

A reference guide for people who think in state machines, simulation loops, and pure functions — and who find event-driven UI confusing.

---

## 1. The Core Mental Model

The single most important idea in modern React:

```
UI = f(state)
```

Your component tree is a **pure function from state to a description of the screen**. You never manually manipulate the UI. You update state, and React recomputes the UI for you.

This is structurally identical to a simulation loop:

```python
# Physics sim
while running:
    state = update(state, inputs)
    render(state)
```

The only difference: React's loop is **event-driven** rather than time-driven. Every user interaction, API response, or timer is a trigger that says "here's new input, recompute the state, re-render."

### The Dataflow

```
[user interaction] → [state update] → [re-render] → [new UI]
                                            ↑
                                       (pure fn of state)
```

### Props Down, Events Up

Data flows from parents to children via **props**. Children signal changes by calling **callback functions** their parents passed down. This gives you a strict, traceable, acyclic data flow — even though the triggers are events.

### Reading Tip

When you're lost in React code, anchor with two questions:

1. **What is the state?** (find all `useState`, Redux slices, server data)
2. **What renders as a function of that state?** (trace the render path top-down)

Event handlers are just the state-transition functions — they're not the main logic.

---

## 2. The Three Layers: JavaScript, TypeScript, React

A common source of confusion is not knowing which layer a given piece of code belongs to.

| Layer | What it is |
|---|---|
| **JavaScript** | The base language that runs in the browser. |
| **TypeScript** | JavaScript + type annotations. Compiles to plain JS before running. Adds nothing at runtime. |
| **JSX** | A syntax extension that lets you write HTML-like tags in JS. Compiles to `React.createElement(...)` calls. |
| **React** | A JavaScript library (functions and objects you import). `useState`, `useEffect`, components, etc. |

### File Extensions

```
.js   — JavaScript
.ts   — TypeScript
.jsx  — JavaScript + JSX (React)
.tsx  — TypeScript + JSX (React)
```

Most React+TypeScript projects use `.tsx` for everything that has JSX, and `.ts` for files that don't (like pure API clients or Redux slices).

### Node.js

JavaScript was originally browser-only. **Node.js** is a runtime that lets JS run outside the browser (like Python). In a React project, Node is used for:

- The **build toolchain** (compiles TSX → JS, bundles files)
- **npm** (the package manager, similar to pip)

The browser never sees Node. Node is purely a development-time tool. The final output is plain HTML/CSS/JS that runs in the browser.

### Decision Table: Which Layer Am I Looking At?

| Confused by... | Layer | Where to look |
|---|---|---|
| `?.`, `??`, `...`, `=>` | JavaScript | MDN JS docs |
| `useState`, `useEffect`, JSX behavior | React | React docs |
| `: string`, `interface`, `<T>` | TypeScript | TS docs (often skippable when reading for logic) |
| `npm install`, build errors | Node / tooling | Project README |

---

## 3. JavaScript & JSX Syntax Survival Guide

### Arrow Functions

Three ways to write a function. The third (arrow) is most common in React.

```js
function add(a, b) { return a + b; }           // classic
const add = function(a, b) { return a + b; };  // function expression
const add = (a, b) => { return a + b; };       // arrow
const add = (a, b) => a + b;                   // arrow with implicit return
```

**Gotcha**: With curly braces, you must `return` explicitly. Without braces, the expression is auto-returned.

```js
const double = x => x * 2;        // returns x * 2
const double = x => { x * 2 };    // returns undefined — common bug
```

### Destructuring

JavaScript lets you unpack objects and arrays directly. React relies on this heavily.

**Object destructuring:**
```js
const node = { id: 1, label: 'foo' };
const { id, label } = node;       // unpacks node.id and node.label
```

This is why hook returns look the way they do:
```js
const { data, isLoading } = useQuery(...);
// useQuery returns ONE object; we're unpacking two properties from it
```

**Array destructuring:**
```js
const [count, setCount] = useState(0);
// useState returns a two-element array; element 0 → count, element 1 → setCount
// the names are yours to choose
```

### Spread Operator (`...`)

Copies and extends objects or arrays:

```js
const newState = { ...oldState, query: 'new' };  // copy oldState, overwrite query
<SearchBar {...props} />                          // pass all props through
```

### JSX: Not HTML

```jsx
const el = <div className="foo">Hello {name}</div>;
// compiles to:
const el = React.createElement('div', { className: 'foo' }, 'Hello ', name);
```

Rules to remember:
- Attributes are camelCase: `className` (not `class`), `onClick` (not `onclick`)
- `{...}` inside JSX = "evaluate this JS expression"
- Capital letter = component (`<SearchBar />`), lowercase = HTML element (`<div />`)

### Callback Props: The `()` Matters

```jsx
<button onClick={handleClick}>      // pass the function
<button onClick={handleClick()}>    // CALL it now, pass the return value — usually a bug
<button onClick={() => handleClick(arg)}>  // wrap in arrow to pass args
```

### Reading a Component Top to Bottom

```jsx
function ResultsList({ results, onSelect, isLoading }) {  // 1. props in (destructured)
    const hasResults = results.length > 0;                 // 2. derived values
    
    if (isLoading) return <div>Loading...</div>;           // 3. early returns
    
    return (                                                // 4. main render — what comes out
        <ul>
            {results.map(node => (                          // JS expression inside JSX
                <li key={node.id} onClick={() => onSelect(node.id)}>
                    {node.label}
                </li>
            ))}
        </ul>
    );
}
```

**Frame**: function signature = what comes in, return = what comes out, everything in between computes the output.

---

## 4. Components

A component is a function that takes props and returns JSX.

```jsx
function SearchBar(props) {
    return <input value={props.query} />;
}

// Equivalent, more common:
function SearchBar({ query, onChange }) {
    return <input value={query} onChange={e => onChange(e.target.value)} />;
}
```

### Using Components

```jsx
<SearchBar query={query} onChange={onQueryChange} />
```

Read this as: *call `SearchBar` with the props object `{ query: query, onChange: onQueryChange }`*.

### Composition

Parents pass data **down** via props and pass **callbacks** down too. Children call those callbacks to signal events upward. This is the entire mechanism for component communication.

### Capital Letters Matter

JSX uses the case of the tag to distinguish HTML elements from your components:
- `<div />` — HTML element
- `<SearchBar />` — your component

---

## 5. Hooks: The Interface to React's Runtime

A **hook** is a function call that reaches into React's runtime during a render. The `use` prefix marks it as a hook.

### What React's Runtime Actually Is

React maintains:

1. A **persistent record** of each component instance, including its state, refs, effects, etc.
2. A **reconciliation loop** that re-runs components when state changes, diffs the output, and patches the DOM.

So the "real" model is:

```
UI = f(state)              ← conceptual
DOM = patch(DOM, f(state)) ← what actually happens
```

You write code as if it's the first; React optimizes it into the second.

### The Slot-Based Model

For each component, React keeps a list of "slots" — one per hook call, in order:

```jsx
function MyComponent() {
    const [count, setCount] = useState(0);   // slot 0
    const myRef = useRef(null);              // slot 1
    useEffect(() => { ... }, [count]);       // slot 2
}
```

Internally:
```
slot 0: { type: state, value: 0, setter: fn }
slot 1: { type: ref, current: null }
slot 2: { type: effect, fn: ..., deps: [0] }
```

When you call `useState()`, you're reading slot 0 for this component instance. This is why **hooks must be called in the same order every render** — no conditionals, no loops, no early returns before hook calls. React identifies them purely by position.

---

## 6. The Core Hooks

### `useState`

Persistent state that **triggers re-renders** when changed.

```jsx
const [count, setCount] = useState(0);
setCount(count + 1);          // re-renders
setCount(c => c + 1);         // functional form — safer for rapid updates
```

### `useRef`

A persistent **mutable box** that does **not** trigger re-renders.

```jsx
const myRef = useRef(initialValue);
myRef.current;                  // read
myRef.current = newValue;       // write (silent — no re-render)
```

Two main use cases:

1. **DOM references**: attach to JSX to get the underlying DOM node.
   ```jsx
   <input ref={inputRef} />
   useEffect(() => inputRef.current.focus(), []);
   ```
2. **Persisted bookkeeping**: timer IDs, frame counters, previous values — anything that needs to survive renders but isn't displayed.

**Don't render `ref.current` in JSX** — it won't update. Use state for displayed values.

### `useEffect`

Runs a **side effect after render**, when dependencies change.

```jsx
useEffect(() => {
    fetchData(param);
}, [param]);   // re-runs only when param changes
```

Read it as: *"after rendering, if `param` changed, run this function."* Not an event handler — a reactive trigger on derived state.

The dependency array:
- `[]` — run once after first render only
- `[a, b]` — run when `a` or `b` changes
- *(omitted)* — run after every render (usually a bug)

### `useMemo` / `useCallback`

Memoize values and functions to avoid recomputation. `useCallback(fn, deps)` is just sugar for `useMemo(() => fn, deps)`.

```jsx
const expensive = useMemo(() => compute(a, b), [a, b]);
const handler = useCallback(() => doSomething(a), [a]);
```

**When to actually use them:**
- `useMemo` — for genuinely expensive computations, or values used as deps elsewhere
- `useCallback` — when a function is passed to a `React.memo`-wrapped child

Don't add these preemptively. Reach for them when you have a real performance problem or referential-stability requirement.

### Hook Cheat-Sheet

| Hook | One-line summary |
|---|---|
| `useState` | Persistent variable; setter triggers re-render |
| `useRef` | Persistent mutable box; changes are silent |
| `useEffect` | Side effect after render, on dependency change |
| `useMemo` | Memoize a computed value |
| `useCallback` | Memoize a function |
| `useContext` | Read from a Context provider (see §10) |
| `useSelector` | Read from Redux store + subscribe |
| `useDispatch` | Get the Redux dispatch function |
| `useQuery` | Fetch + cache server data (TanStack Query) |
| `useMutation` | Send writes to the backend (TanStack Query) |

---

## 7. External Boundaries

The two places your app touches the outside world.

### User Input: Events as State Transitions

Event handlers are plain callbacks whose job is to **update state**.

```jsx
<button onClick={() => setCount(count + 1)}>Click</button>
```

Input comes in → state updates → React re-renders. The callback is just the `∂state/∂t` equation.

### Backend API Calls: The Four-State Lifecycle

Async calls have an implicit state machine:

```
idle → loading → success
                → error
```

The naive pattern:

```jsx
const [data, setData] = useState(null);
const [loading, setLoading] = useState(false);
const [error, setError] = useState(null);

useEffect(() => {
    setLoading(true);
    fetch('/api/nodes')
        .then(r => r.json())
        .then(d => { setData(d); setLoading(false); })
        .catch(e => { setError(e); setLoading(false); });
}, []);
```

This works but is verbose. **TanStack Query** manages this state machine for you.

### TanStack Query: Reads

```jsx
const { data, isLoading, isError } = useQuery({
    queryKey: ['nodes', 'search', query],
    queryFn: () => searchNodes(query),
    enabled: query.length > 2,
    staleTime: 30_000,
});
```

- `queryKey`: cache key (array; structured as a namespace hierarchy)
- `queryFn`: the async function that does the actual fetching
- `enabled`: only run when this is true
- `staleTime`: how long to consider cached data fresh

**Cache keys as namespace:**
```js
['nodes']                       // everything about nodes
['nodes', 'search']             // just search results
['nodes', 'search', query]      // results for a specific query
```

`queryClient.invalidateQueries(['nodes'])` invalidates everything under `'nodes'`. `invalidateQueries(['nodes', 'search'])` invalidates only search results. Coarse or fine, your choice.

### TanStack Query: Writes (Mutations)

```jsx
const mutation = useMutation({
    mutationFn: (newNode) => fetch('/api/nodes', {
        method: 'POST',
        body: JSON.stringify(newNode),
    }),
    onSuccess: () => queryClient.invalidateQueries(['nodes']),
});

mutation.mutate({ label: 'new node' });
```

The pattern: **send the write, invalidate the cache, let the read pipeline refresh automatically.** You never manually patch local state to reflect a write.

---

## 8. State Management with Redux

### When to Use What

| Type of state | Where it lives |
|---|---|
| UI state shared across components (selected item, open panels, query string) | **Redux** |
| Server data (results, fetched detail) | **TanStack Query** |
| Local-only state (input focus, hover, this component's expanded state) | **`useState`** in the component |
| Derived values | **`useMemo`** or computed inline |

### Redux Toolkit Slices

A slice bundles state, reducers, and action creators:

```js
import { createSlice } from '@reduxjs/toolkit';

const searchSlice = createSlice({
    name: 'search',
    initialState: { query: '', selectedNodeId: null },
    reducers: {
        setQuery(state, action) {
            state.query = action.payload;
        },
        selectNode(state, action) {
            state.selectedNodeId = action.payload;
        },
    },
});

export const { setQuery, selectNode } = searchSlice.actions;
export default searchSlice.reducer;
```

### Actions and `action.payload`

When you call `dispatch(setQuery('hello'))`, Redux Toolkit constructs:

```js
{ type: 'search/setQuery', payload: 'hello' }
```

- `type`: string identifying which reducer case to run
- `payload`: the data carried with the action (convention, not a JS feature)

`setQuery` is an **action creator** — a function that builds the action object. The reducer receives `(state, action)` and reads `action.payload`.

### What `createSlice` Returns

```js
searchSlice.actions    // { setQuery, selectNode } — auto-generated action creators
searchSlice.reducer    // a single reducer function for this slice
```

The exports at the bottom of the file:

```js
export const { setQuery, selectNode } = searchSlice.actions;  // named exports
export default searchSlice.reducer;                            // default export
```

### Reading and Dispatching from Components

```jsx
import { useSelector, useDispatch } from 'react-redux';

const query = useSelector(state => state.search.query);  // read + subscribe
const dispatch = useDispatch();
dispatch(setQuery('hello'));                              // write
```

`useSelector` takes a selector function `state => value`. Redux calls it on every store update; if the returned value changed, the component re-renders.

---

## 9. Custom Hooks

A **custom hook is just a function that calls other hooks**. The `use` prefix is a convention that signals "hook rules apply here" to React's linting.

### Why Custom Hooks

To encapsulate a coherent slice of reactive logic and hide it behind a clean interface — same reason you'd extract a class or module in backend code.

### Example

```js
function useNodeSearch() {
    const dispatch = useDispatch();
    const query = useSelector(state => state.search.query);
    
    const { data: results, isLoading } = useQuery({
        queryKey: ['nodes', 'search', query],
        queryFn: () => searchNodes(query),
        enabled: query.length > 2,
    });
    
    return {
        query,
        results: results ?? [],
        isLoading,
        onQueryChange: (q) => dispatch(setQuery(q)),
    };
}
```

The component using it doesn't need to know whether the data comes from Redux, TanStack Query, or local state.

### Common Shapes

| Shape | Example |
|---|---|
| **State + actions** | `useChat` — owns a state object, exposes operations |
| **Data fetching** | `useNodeDetail(id)` — wraps `useQuery` with app-specific defaults |
| **Derived values** | `useFilteredResults(results, filters)` — heavy `useMemo` internally |
| **DOM / browser** | `useResizeObserver`, `useKeyboardShortcut` — hides refs and effects |

---

## 10. Things We Haven't Covered Yet

### `useContext` — Sharing State Without Prop Drilling

When you need to share a value across many components but it doesn't fit Redux (e.g. theme, current user, locale), use Context.

```jsx
const ThemeContext = React.createContext('light');

// Provider wraps the subtree
<ThemeContext.Provider value="dark">
    <App />
</ThemeContext.Provider>

// Any descendant reads it
function Button() {
    const theme = useContext(ThemeContext);
    return <button className={theme}>Click</button>;
}
```

Use Context for slow-changing, app-wide values. Don't use it as a Redux replacement for frequently-updating state — every consumer re-renders when the value changes.

### Conditional Rendering

JSX is just an expression, so use normal JS conditionals:

```jsx
// Ternary — common
{isLoading ? <Spinner /> : <Results data={data} />}

// && — when there's no "else" branch
{error && <ErrorMessage error={error} />}

// Early return — when the whole component branches
if (!data) return null;
return <Results data={data} />;
```

### Lists and Keys

Render an array of items by mapping it to JSX. The `key` prop is **required** and must be **stable and unique** within the list:

```jsx
{results.map(node => (
    <li key={node.id}>{node.label}</li>
))}
```

React uses `key` to identify items across renders. Don't use array indices as keys if the list can reorder — that confuses React's diffing.

### Controlled vs Uncontrolled Inputs

**Controlled** (the React way) — the input's value is held in state:

```jsx
const [text, setText] = useState('');
<input value={text} onChange={e => setText(e.target.value)} />
```

The component owns the value. The input always displays exactly what state says.

**Uncontrolled** — the DOM owns the value, you read it via a ref:

```jsx
const inputRef = useRef(null);
<input ref={inputRef} defaultValue="" />
// later: inputRef.current.value
```

Use controlled by default. Use uncontrolled for cases where React doesn't need to react to every keystroke (e.g. simple form submission).

---

## 11. Worked Example

A search interface backed by FastAPI/Weaviate, with results and a detail panel.

### File Structure

```
src/
  store/
    index.js              # Redux store setup
    searchSlice.js        # UI state: query, selected node
  hooks/
    useNodeSearch.js      # custom hook: search logic
  components/
    SearchBar.jsx         # controlled input
    ResultsList.jsx       # renders results
    NodeDetail.jsx        # fetches and renders selected node
  pages/
    SearchPage.jsx        # composes everything
  api/
    nodes.js              # raw async fetch functions
```

**Principle**: React concerns in hooks/components, API concerns in `api/`, global UI state in `store/`.

### `api/nodes.js` — Plain Async Functions

```js
const BASE = 'https://api.myapp.com';

export async function searchNodes(query) {
    const res = await fetch(`${BASE}/nodes/search?q=${encodeURIComponent(query)}`);
    if (!res.ok) throw new Error('Search failed');
    return res.json();
}

export async function fetchNodeDetail(id) {
    const res = await fetch(`${BASE}/nodes/${id}`);
    if (!res.ok) throw new Error('Failed to fetch node');
    return res.json();
}
```

No React. Easy to test in isolation.

### `store/searchSlice.js` — UI State

```js
import { createSlice } from '@reduxjs/toolkit';

const searchSlice = createSlice({
    name: 'search',
    initialState: { query: '', selectedNodeId: null },
    reducers: {
        setQuery(state, action) {
            state.query = action.payload;
            state.selectedNodeId = null;  // clear selection on new search
        },
        selectNode(state, action) {
            state.selectedNodeId = action.payload;
        },
    },
});

export const { setQuery, selectNode } = searchSlice.actions;
export default searchSlice.reducer;
```

### `hooks/useNodeSearch.js` — Custom Hook

```js
import { useSelector, useDispatch } from 'react-redux';
import { useQuery } from '@tanstack/react-query';
import { setQuery, selectNode } from '../store/searchSlice';
import { searchNodes } from '../api/nodes';

export function useNodeSearch() {
    const dispatch = useDispatch();
    const query = useSelector(state => state.search.query);
    const selectedNodeId = useSelector(state => state.search.selectedNodeId);

    const { data: results, isLoading, isError } = useQuery({
        queryKey: ['nodes', 'search', query],
        queryFn: () => searchNodes(query),
        enabled: query.length > 2,
        staleTime: 30_000,
    });

    return {
        query,
        results: results ?? [],
        isLoading,
        isError,
        selectedNodeId,
        onQueryChange: (q) => dispatch(setQuery(q)),
        onSelectNode: (id) => dispatch(selectNode(id)),
    };
}
```

### Components

```jsx
// SearchBar.jsx — owns nothing, pure function of props
export function SearchBar({ query, onChange }) {
    return (
        <input
            value={query}
            onChange={e => onChange(e.target.value)}
            placeholder="Search nodes..."
        />
    );
}

// ResultsList.jsx — also owns nothing
export function ResultsList({ results, selectedNodeId, onSelect, isLoading, isError }) {
    if (isLoading) return <div>Loading...</div>;
    if (isError)   return <div>Something went wrong.</div>;
    if (!results.length) return <div>No results.</div>;

    return (
        <ul>
            {results.map(node => (
                <li
                    key={node.id}
                    onClick={() => onSelect(node.id)}
                    style={{ fontWeight: node.id === selectedNodeId ? 'bold' : 'normal' }}
                >
                    {node.label}
                </li>
            ))}
        </ul>
    );
}

// NodeDetail.jsx — owns its own fetch (local display concern)
import { useQuery } from '@tanstack/react-query';
import { fetchNodeDetail } from '../api/nodes';

export function NodeDetail({ nodeId }) {
    const { data: node, isLoading } = useQuery({
        queryKey: ['nodes', 'detail', nodeId],
        queryFn: () => fetchNodeDetail(nodeId),
        enabled: !!nodeId,
    });

    if (!nodeId)   return <div>Select a node to see detail.</div>;
    if (isLoading) return <div>Loading detail...</div>;
    if (!node)     return null;

    return (
        <div>
            <h2>{node.label}</h2>
            <p>{node.description}</p>
        </div>
    );
}
```

### `pages/SearchPage.jsx` — Composition

```jsx
import { useNodeSearch } from '../hooks/useNodeSearch';
import { SearchBar } from '../components/SearchBar';
import { ResultsList } from '../components/ResultsList';
import { NodeDetail } from '../components/NodeDetail';

export function SearchPage() {
    const {
        query, results, isLoading, isError,
        selectedNodeId, onQueryChange, onSelectNode,
    } = useNodeSearch();

    return (
        <div style={{ display: 'flex', gap: '2rem' }}>
            <div style={{ flex: 1 }}>
                <SearchBar query={query} onChange={onQueryChange} />
                <ResultsList
                    results={results}
                    selectedNodeId={selectedNodeId}
                    onSelect={onSelectNode}
                    isLoading={isLoading}
                    isError={isError}
                />
            </div>
            <div style={{ flex: 2 }}>
                <NodeDetail nodeId={selectedNodeId} />
            </div>
        </div>
    );
}
```

### Full Data Flow Trace

**User types in SearchBar:**
```
onChange → onQueryChange → dispatch(setQuery(q))
    → Redux state.search.query updates
    → useNodeSearch re-runs (subscribed via useSelector)
    → useQuery sees new queryKey → fires searchNodes(query)
    → results arrive → useNodeSearch returns new results
    → SearchPage re-renders → ResultsList renders new results
```

**User clicks a result:**
```
onClick → onSelectNode → dispatch(selectNode(id))
    → Redux state.search.selectedNodeId updates
    → SearchPage re-renders → NodeDetail receives new nodeId
    → NodeDetail's useQuery fires fetchNodeDetail(id)
    → Detail data arrives → NodeDetail renders
```

Every arrow is a state transition. The UI at any moment is just a function of the current state.

### Why `searchNodes` Lives in the Hook but `fetchNodeDetail` Doesn't

| Concern | Where it lives | Why |
|---|---|---|
| Search | In `useNodeSearch` | Shared concern: driven by user input, lives in Redux, feeds results to multiple components |
| Node detail | In `NodeDetail` directly | Local display concern: only one component uses it, lifecycle tied to component visibility |

**General principle**: state and data live as close to where they're used as possible, and no higher. Hoist only when something is genuinely shared.

---

## 12. Quick Reference

### Architectural Heuristics

| Need | Use |
|---|---|
| Shared UI state (selection, filters, modes) | Redux |
| Server data | TanStack Query (`useQuery`) |
| Local component state | `useState` |
| Persistent value that shouldn't trigger renders | `useRef` |
| Side effect on state change | `useEffect` |
| Coherent reactive logic reused or simplified | Custom hook |
| App-wide slow-changing value (theme, user) | `useContext` |
| Expensive computation | `useMemo` |

### "Where Should This Data Fetch Live?"

```
Is the data shared across components, or driven by app-level user input?
    YES → custom hook (or page-level useQuery)
    NO  → useQuery inside the component that needs it
```

### Common Syntax Patterns at a Glance

```js
// Destructure object — pull properties by name
const { a, b } = obj;

// Destructure array — pull by position
const [first, second] = arr;

// Spread — copy and extend
const newObj = { ...old, key: 'new' };
const newArr = [...old, item];

// Arrow function
const fn = (a, b) => a + b;

// Optional chaining — safe property access
obj?.foo?.bar              // undefined if obj or foo is null/undefined

// Nullish coalescing — fallback for null/undefined only
results ?? []              // results if defined, else []
```

### JSX Patterns at a Glance

```jsx
// Embed a JS expression
<div>{value}</div>

// Conditional rendering
{cond ? <A /> : <B />}
{cond && <A />}

// List rendering (key required)
{items.map(item => <li key={item.id}>{item.label}</li>)}

// Event handler
<button onClick={() => doThing()}>Click</button>

// Pass props
<Child name="foo" onSomething={handler} />

// Spread props
<Child {...props} />
```

### Hook Recap

| Hook | Returns | Triggers re-render on change? |
|---|---|---|
| `useState(init)` | `[value, setter]` | Yes |
| `useRef(init)` | `{ current: value }` | No |
| `useEffect(fn, deps)` | nothing | n/a (runs after render) |
| `useMemo(fn, deps)` | computed value | n/a (returns cached value) |
| `useCallback(fn, deps)` | the function | n/a (returns stable reference) |
| `useContext(Ctx)` | current context value | Yes (when provider updates) |
| `useSelector(fn)` | value from Redux | Yes (when selected value changes) |
| `useDispatch()` | dispatch function | n/a |
| `useQuery(opts)` | `{ data, isLoading, ... }` | Yes (when query state changes) |
| `useMutation(opts)` | `{ mutate, isLoading, ... }` | Yes (when mutation state changes) |

### The Single Most Important Sentence

**The UI is a pure function of state. Events update state. State changes trigger re-renders. Everything else is mechanism.**
