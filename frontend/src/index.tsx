import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'

// Register PWA
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register(new URL('./sw.ts', import.meta.url), { type: 'module' })
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
