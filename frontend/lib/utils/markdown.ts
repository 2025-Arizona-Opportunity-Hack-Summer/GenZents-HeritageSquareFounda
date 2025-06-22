import { marked } from 'marked'
import DOMPurify from 'isomorphic-dompurify'

// Configure marked for better security and performance
marked.setOptions({
  breaks: true,
  gfm: true,
})

export const renderMarkdown = (content: string): string => {
  const html = marked.parse(content) as string
  
  return DOMPurify.sanitize(html, {
    ALLOWED_TAGS: [
      'p', 'br', 'strong', 'em', 'u', 's', 'code', 'pre',
      'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
      'ul', 'ol', 'li',
      'blockquote',
      'a',
      'table', 'thead', 'tbody', 'tr', 'th', 'td'
    ],
    ALLOWED_ATTR: ['href', 'title', 'target', 'rel', 'class'],
    FORBID_ATTR: ['style', 'onclick', 'onload', 'onerror'],
    FORBID_TAGS: ['script', 'object', 'embed', 'base', 'meta'],
    ALLOW_DATA_ATTR: false,
  })
}
