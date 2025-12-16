import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import ImageUploader from './components/ImageUploader';
import EditorControls from './components/EditorControls';
import ResultDisplay from './components/ResultDisplay';
import { ImageFile, AppStatus, GeneratedImage } from './types';
import { generateChristmasEdit } from './services/geminiService';
import { XCircle, Sparkles, KeyRound, ChevronRight } from 'lucide-react';

const App: React.FC = () => {
  const [status, setStatus] = useState<AppStatus>(AppStatus.IDLE);
  const [selectedImage, setSelectedImage] = useState<ImageFile | null>(null);
  const [generatedImage, setGeneratedImage] = useState<GeneratedImage | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  // API Key State
  const [hasApiKey, setHasApiKey] = useState<boolean>(false);
  const [isCheckingKey, setIsCheckingKey] = useState<boolean>(true);

  useEffect(() => {
    checkApiKey();
  }, []);

  const checkApiKey = async () => {
    if ((window as any).aistudio) {
      const hasKey = await (window as any).aistudio.hasSelectedApiKey();
      setHasApiKey(hasKey);
    } else {
      // Fallback for local dev if window.aistudio isn't present
      setHasApiKey(true);
    }
    setIsCheckingKey(false);
  };

  const handleSelectKey = async () => {
    if ((window as any).aistudio) {
      try {
        await (window as any).aistudio.openSelectKey();
        // Assume success or check again
        const hasKey = await (window as any).aistudio.hasSelectedApiKey();
        setHasApiKey(hasKey);
      } catch (e) {
        console.error("Failed to select key", e);
      }
    }
  };

  const handleImageSelected = (image: ImageFile) => {
    setSelectedImage(image);
    setGeneratedImage(null);
    setStatus(AppStatus.IDLE);
    setError(null);
  };

  const handleReset = () => {
    setSelectedImage(null);
    setGeneratedImage(null);
    setStatus(AppStatus.IDLE);
    setError(null);
  };

  const handleGenerate = async (prompt: string) => {
    if (!selectedImage) return;

    setStatus(AppStatus.LOADING);
    setError(null);

    try {
      const result = await generateChristmasEdit(
        selectedImage.base64,
        selectedImage.mimeType,
        prompt
      );
      setGeneratedImage(result);
      setStatus(AppStatus.SUCCESS);
    } catch (err: any) {
      console.error(err);
      
      // If we get an entity not found error, it might be the key. Reset key state.
      if (err.message && (err.message.includes('Requested entity was not found') || err.message.includes('Access denied'))) {
        setHasApiKey(false);
        setError('Your API Key may be invalid or not support Gemini 3 Pro. Please select a new key.');
      } else {
        setError(err.message || 'Something went wrong. Please try again.');
      }
      
      setStatus(AppStatus.ERROR);
    }
  };

  if (isCheckingKey) {
    return <div className="min-h-screen flex items-center justify-center bg-xmas-cream text-xmas-dark">Loading...</div>;
  }

  // API Key Selection Landing Page
  if (!hasApiKey) {
    return (
      <div className="min-h-screen pb-20 flex flex-col">
        <Header />
        <main className="flex-grow flex items-center justify-center px-4 py-12">
          <div className="max-w-md w-full bg-white rounded-2xl shadow-xl p-8 border-t-4 border-xmas-gold text-center">
            <div className="w-16 h-16 bg-xmas-red/10 rounded-full flex items-center justify-center mx-auto mb-6">
              <KeyRound className="text-xmas-red" size={32} />
            </div>
            <h2 className="text-2xl font-serif font-bold text-xmas-dark mb-4">
              Unlock the Magic
            </h2>
            <p className="text-gray-600 mb-8 leading-relaxed">
              To create high-quality <strong>Gemini 3 Pro</strong> Christmas edits, please select a paid API key. 
              <br/><br/>
              <span className="text-sm text-gray-500">
                You will need a Google Cloud project with billing enabled.
              </span>
            </p>
            <button
              onClick={handleSelectKey}
              className="w-full bg-xmas-red hover:bg-red-700 text-white font-bold py-4 px-6 rounded-xl shadow-lg transition-all transform hover:scale-[1.02] active:scale-[0.98] flex items-center justify-center gap-2"
            >
              Select API Key <ChevronRight size={20} />
            </button>
            <p className="mt-4 text-xs text-gray-400">
              <a href="https://ai.google.dev/gemini-api/docs/billing" target="_blank" rel="noreferrer" className="underline hover:text-xmas-green">
                Learn more about Gemini API billing
              </a>
            </p>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen pb-20">
      <Header />

      <main className="max-w-6xl mx-auto px-4 py-8">
        {/* Error Notification */}
        {error && (
          <div className="mb-6 p-4 bg-red-100 border-l-4 border-red-500 text-red-700 rounded shadow-md flex items-start gap-3">
            <XCircle className="shrink-0 mt-0.5" size={20} />
            <div>
              <h4 className="font-bold">Ho Ho No!</h4>
              <p>{error}</p>
              {/* If it's an auth error, give them a quick way to reset */}
              {(error.includes('Access denied') || error.includes('API Key')) && (
                <button
                  onClick={handleSelectKey}
                  className="mt-2 text-sm font-bold underline hover:text-red-900"
                >
                  Change API Key
                </button>
              )}
            </div>
          </div>
        )}

        {!selectedImage ? (
          // Initial State: Upload
          <div className="max-w-2xl mx-auto mt-10">
            <div className="bg-white rounded-2xl shadow-xl p-8 border border-xmas-gold/20">
              <div className="text-center mb-8">
                <h2 className="text-3xl font-serif font-bold text-xmas-dark mb-2">Turn your pet into a Holiday Star</h2>
                <p className="text-gray-600">Upload a photo of your dog, cat, or hamster and watch the magic happen.</p>
              </div>
              <ImageUploader onImageSelected={handleImageSelected} />
            </div>
          </div>
        ) : (
          // Editor State
          <div className="flex flex-col lg:flex-row gap-8">
            {/* Left Column: Image Display */}
            <div className="w-full lg:w-2/3 order-2 lg:order-1">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-2xl font-serif font-bold text-xmas-dark">Preview</h2>
                <button
                  onClick={handleReset}
                  className="flex items-center gap-1 text-sm font-medium text-gray-500 hover:text-xmas-red transition-colors group"
                >
                  <Sparkles size={14} className="group-hover:animate-pulse" />
                  <span>Start a new festive tale</span>
                </button>
              </div>
              <ResultDisplay
                originalImage={selectedImage}
                generatedImage={generatedImage}
              />
            </div>

            {/* Right Column: Controls */}
            <div className="w-full lg:w-1/3 order-1 lg:order-2">
              <div className="sticky top-6">
                <EditorControls 
                  onGenerate={handleGenerate} 
                  status={status}
                />
              </div>
            </div>
          </div>
        )}
      </main>
      
      <footer className="fixed bottom-0 w-full bg-xmas-dark text-xmas-cream py-3 text-center text-sm z-50">
        <p>Made with ❤️ & Gemini AI for your furry friends</p>
      </footer>
    </div>
  );
};

export default App;