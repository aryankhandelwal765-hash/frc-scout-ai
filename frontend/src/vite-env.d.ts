/// <reference types="vite/client" />

declare module '*.module.css' {
  const content: Record<string, string>
  export default content
}

declare module '*.css' {
  const content: string
  export default content
}
