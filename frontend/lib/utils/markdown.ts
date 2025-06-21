import { marked } from 'marked'
import DOMPurify from 'isomorphic-dompurify'

export const renderMarkdown = (content: string): string => {
  const html = marked.parse(content, {
    breaks: true,
    gfm: true,
  }) as string
    return DOMPurify.sanitize(html, {
    ALLOWED_TAGS: [
      'p', 'br', 'strong', 'em', 'u', 's', 'code', 'pre',
      'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
      'ul', 'ol', 'li',
      'blockquote',
      'a',
      'table', 'thead', 'tbody', 'tr', 'th', 'td'
    ],
    ALLOWED_ATTR: ['href', 'title', 'target', 'rel'],
    ADD_ATTR: ['target', 'rel']
  })
}
