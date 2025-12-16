# Paw-rfect Christmas 🎄🐾

Transform your pet photos into festive Christmas masterpieces using AI! This application uses the Google Gemini API to magically edit photos while preserving your pet's identity.

## Features

*   **Pet Detection**: Automatically checks if uploaded photos contain real animals.
*   **Magical Edits**: Uses `gemini-3-pro-image-preview` for high-quality, photorealistic Christmas edits.
*   **Style Presets**: Choose from Cozy Fireplace, Winter Wonderland, Santa Helper, and more.
*   **Festive UI**: A beautiful, responsive interface with snow animations.

## Getting Started

Follow these instructions to run the project locally on your machine.

### Prerequisites

*   **Node.js** (version 18 or higher)
*   A **Google Gemini API Key**. Get one at [aistudio.google.com](https://aistudio.google.com/).

### Installation

1.  **Install Dependencies**
    Open your terminal in the project folder and run:
    ```bash
    npm install vite @vitejs/plugin-react react react-dom @google/genai lucide-react
    ```

2.  **Verify Environment**
    Ensure you have a valid Google API key in your environment:
    ```env
    GOOGLE_API_KEY=your_actual_api_key_here
    ```

3.  **Run the App**
    Start the local development server:
    ```bash
    npx vite
    ```

    The app should open automatically in your browser at `http://localhost:3000`.

## Troubleshooting

*   **Error: "Access denied"**: Ensure your API key in `.env` is correct and has billing enabled (required for `gemini-3-pro-image-preview`).
*   **Snow animation missing**: Ensure your internet connection works to load the Tailwind CDN.
