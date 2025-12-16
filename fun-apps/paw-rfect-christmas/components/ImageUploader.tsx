import React, { useRef, useState } from 'react';
import { Upload, Image as ImageIcon, Loader2 } from 'lucide-react';
import { ImageFile } from '../types';

interface ImageUploaderProps {
  onImageSelected: (image: ImageFile) => void;
}

const ImageUploader: React.FC<ImageUploaderProps> = ({ onImageSelected }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      processFile(file);
    }
  };

  const processFile = (file: File) => {
    setIsProcessing(true);

    try {
      // Explicitly reject HEIC since we removed support
      if (file.type.includes('heic') || file.type.includes('heif') || /\.(heic|heif)$/i.test(file.name)) {
        alert("HEIC format is not supported. Please use JPG, PNG or WEBP.");
        setIsProcessing(false);
        return;
      }

      // Basic validation
      if (!file.type.startsWith('image/')) {
        alert(`File type '${file.type}' is not supported. Please upload an image.`);
        setIsProcessing(false);
        return;
      }

      const reader = new FileReader();
      reader.onloadend = () => {
        const result = reader.result as string;
        // Extract base64 clean string (remove data:image/xxx;base64, prefix)
        const base64 = result.split(',')[1];
        
        onImageSelected({
          file: file,
          previewUrl: result,
          base64,
          mimeType: file.type
        });
        setIsProcessing(false);
      };
      reader.onerror = () => {
        alert('Error reading file data.');
        setIsProcessing(false);
      };
      reader.readAsDataURL(file);

    } catch (error: any) {
      console.error("Error processing file:", error);
      alert(`Error processing image: ${error.message}`);
      setIsProcessing(false);
    }
  };

  return (
    <div
      className={`border-2 border-dashed border-xmas-green/40 bg-white/50 rounded-xl p-8 md:p-12 text-center transition-all cursor-pointer group relative
        ${isProcessing ? 'cursor-wait opacity-70' : 'hover:bg-white/80'}`}
      onClick={() => !isProcessing && fileInputRef.current?.click()}
    >
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        className="hidden"
        accept="image/jpeg,image/png,image/webp"
      />

      {isProcessing ? (
        <div className="absolute inset-0 flex flex-col items-center justify-center z-10 bg-white/50 backdrop-blur-sm rounded-xl">
          <Loader2 className="animate-spin text-xmas-green mb-2" size={48} />
          <p className="text-xmas-dark font-medium">Processing Image...</p>
        </div>
      ) : (
        <>
          <div className="w-20 h-20 bg-xmas-green/10 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:bg-xmas-green/20 transition-colors">
            <Upload className="text-xmas-green" size={40} />
          </div>
          <h3 className="text-xl font-bold text-xmas-dark mb-2">Upload your Pet Photo</h3>
          <p className="text-gray-600 mb-6">Click to browse or drop your file here</p>
          <div className="flex flex-wrap justify-center gap-4 text-sm text-gray-500">
            <span className="flex items-center gap-1"><ImageIcon size={14} /> PNG</span>
            <span className="flex items-center gap-1"><ImageIcon size={14} /> JPG</span>
            <span className="flex items-center gap-1"><ImageIcon size={14} /> WEBP</span>
          </div>
        </>
      )}
    </div>
  );
};

export default ImageUploader;