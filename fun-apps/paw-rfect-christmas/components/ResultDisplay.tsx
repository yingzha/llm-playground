import React, { useState } from 'react';
import { Download, ArrowRight, Eye } from 'lucide-react';
import { ImageFile, GeneratedImage } from '../types';

interface ResultDisplayProps {
  originalImage: ImageFile;
  generatedImage: GeneratedImage | null;
}

const ResultDisplay: React.FC<ResultDisplayProps> = ({ originalImage, generatedImage }) => {
  const [viewMode, setViewMode] = useState<'side' | 'toggle'>('side');

  const handleDownload = () => {
    if (!generatedImage) return;
    const link = document.createElement('a');
    link.href = `data:${generatedImage.mimeType};base64,${generatedImage.base64}`;
    link.download = `xmas-pet-${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-6">
      {/* View Toggle for Mobile */}
      <div className="md:hidden flex justify-center mb-4">
        <button
          onClick={() => setViewMode(viewMode === 'side' ? 'toggle' : 'side')}
          className="flex items-center gap-2 bg-white px-4 py-2 rounded-full shadow text-sm font-medium text-xmas-dark"
        >
          <Eye size={16} /> {viewMode === 'side' ? 'Switch to Stacked' : 'Switch to Side-by-Side'}
        </button>
      </div>

      <div className={`grid gap-6 ${viewMode === 'side' ? 'md:grid-cols-2' : 'grid-cols-1'}`}>
        {/* Original */}
        <div className="bg-white p-3 rounded-xl shadow-lg transform rotate-[-1deg] hover:rotate-0 transition-transform duration-300">
          <div className="bg-gray-100 rounded-lg overflow-hidden aspect-square relative group">
            <img
              src={originalImage.previewUrl}
              alt="Original"
              className="w-full h-full object-cover"
            />
            <div className="absolute top-3 left-3 bg-black/50 text-white px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider backdrop-blur-sm">
              Original
            </div>
          </div>
        </div>

        {/* Generated */}
        <div className="relative">
          {generatedImage ? (
            <div className="bg-white p-3 rounded-xl shadow-xl border-2 border-xmas-gold transform rotate-[1deg] hover:rotate-0 transition-transform duration-300 h-full">
              <div className="bg-xmas-red/5 rounded-lg overflow-hidden aspect-square relative group h-full">
                <img
                  src={`data:${generatedImage.mimeType};base64,${generatedImage.base64}`}
                  alt="Xmas Version"
                  className="w-full h-full object-cover"
                />
                <div className="absolute top-3 left-3 bg-xmas-red text-white px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider shadow-md">
                  Xmas Magic
                </div>

                {/* Download Overlay */}
                <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors flex items-center justify-center opacity-0 group-hover:opacity-100">
                  <button
                    onClick={handleDownload}
                    className="bg-white text-xmas-dark font-bold py-3 px-6 rounded-full shadow-lg flex items-center gap-2 hover:bg-xmas-gold transition-colors transform translate-y-4 group-hover:translate-y-0 duration-300"
                  >
                    <Download size={20} /> Download
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="h-full min-h-[300px] flex flex-col items-center justify-center border-2 border-dashed border-gray-300 rounded-xl bg-gray-50 text-gray-400 p-8 text-center">
              <div className="w-16 h-16 bg-gray-200 rounded-full flex items-center justify-center mb-4">
                <ArrowRight size={32} className="text-gray-400" />
              </div>
              <p>Your festive masterpiece will appear here</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ResultDisplay;