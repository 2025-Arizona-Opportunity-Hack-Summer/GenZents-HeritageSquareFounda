import { Version } from "../../lib/types/version"

export const sampleVersions: Version[] = [
  {
    id: "v1.3.0",
    version: "1.3.0",
    title: "Enhanced Chat Interface",
    description: "Added voice input, file attachments, and improved message formatting",
    author: "John Doe",
    date: new Date("2024-01-15T10:30:00"),
    branch: "main",
    tag: "stable",
    status: "current",
    changes: [
      {
        id: "1",
        type: "added",
        file: "components/chat/voice-input.tsx",
        description: "Added voice input functionality",
        linesAdded: 120,
        linesRemoved: 0,
      },
      {
        id: "2",
        type: "modified",
        file: "components/chat/chat-input.tsx",
        description: "Enhanced input with file attachments",
        linesAdded: 45,
        linesRemoved: 12,
      },
    ],
  },
  {
    id: "v1.2.1",
    version: "1.2.1",
    title: "Bug Fixes and Performance",
    description: "Fixed memory leaks and improved response times",
    author: "Jane Smith",
    date: new Date("2024-01-10T14:20:00"),
    branch: "main",
    status: "previous",
    changes: [
      {
        id: "3",
        type: "modified",
        file: "components/providers/chat-provider.tsx",
        description: "Fixed memory leak in message handling",
        linesAdded: 8,
        linesRemoved: 15,
      },
    ],
  },
  {
    id: "v1.2.0",
    version: "1.2.0",
    title: "Theme System",
    description: "Added dark mode and custom themes",
    author: "Bob Wilson",
    date: new Date("2024-01-05T09:15:00"),
    branch: "main",
    tag: "feature",
    status: "previous",
    changes: [
      {
        id: "4",
        type: "added",
        file: "components/theme-provider.tsx",
        description: "Added theme system",
        linesAdded: 200,
        linesRemoved: 0,
      },
    ],
  },
]
