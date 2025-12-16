import React, { useState } from 'react';
import { Wand2, RefreshCw } from 'lucide-react';
import { PRESET_STYLES } from '../constants';
import { AppStatus } from '../types';

interface EditorControlsProps {
  onGenerate: (prompt: string) => void;
  status: AppStatus;
}

const EditorControls: React.FC<EditorControlsProps> = ({ onGenerate, status }) => {
  const [customPrompt, setCustomPrompt] = useState('');
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);

  const handleGenerate = () => {
    let finalPrompt = customPrompt;
    if (selectedPreset) {
      const preset = PRESET_STYLES.find(p => p.id === selectedPreset);
      if (preset) {
        finalPrompt = finalPrompt ? `${preset.prompt} Also: ${finalPrompt}` : preset.prompt;
      }
    }
    onGenerate(finalPrompt);
  };

  const handlePresetClick = (id: string) => {
    if (selectedPreset === id) {
      setSelectedPreset(null);
    } else {
      setSelectedPreset(id);
    }
  };

  const isLoading = status === AppStatus.LOADING;

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200 h-full flex flex-col">
      <h2 className="text-xl font-bold text-xmas-dark mb-4 flex items-center gap-2">
        <Wand2 className="text-xmas-gold" />
        Customize Style
      </h2>
      
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">Choose a Vibe</label>
        <div className="grid grid-cols-2 gap-2">
          {PRESET_STYLES.map((preset) => (
            <button
              key={preset.id}
              onClick={() => handlePresetClick(preset.id)}
              className={`p-3 rounded-lg border text-left transition-all flex items-center gap-2
                ${selectedPreset === preset.id 
                  ? 'border-xmas-red bg-red-50 text-xmas-red ring-1 ring-xmas-red' 
                  : 'border-gray-200 hover:border-xmas-green hover:bg-green-50'
                }`}
            >
              <span className="text-lg">{preset.icon}</span>
              <span className="text-sm font-medium">{preset.label}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="mb-6 flex-grow">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Your Wishes (Optional)
        </label>
        <textarea
          value={customPrompt}
          onChange={(e) => setCustomPrompt(e.target.value)}
          placeholder="e.g. Make my dog look like a reindeer, add a snowy window..."
          className="w-full h-32 p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-xmas-green focus:border-transparent resize-none"
        />
      </div>

      <button
        onClick={handleGenerate}
        disabled={isLoading}
        className={`w-full py-4 rounded-xl text-white font-bold text-lg shadow-md transition-all flex items-center justify-center gap-2
          ${isLoading 
            ? 'bg-gray-400 cursor-not-allowed' 
            : 'bg-xmas-red hover:bg-red-700 hover:scale-[1.02] active:scale-[0.98]'
          }`}
      >
        {isLoading ? (
          <>
            <RefreshCw className="animate-spin" />
            Creating Magic...
          </>
        ) : (
          <>
            <Wand2 />
            Xmasify My Pet!
          </>
        )}
      </button>
    </div>
  );
};

export default EditorControls;