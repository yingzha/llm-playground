export enum AppStatus {
  IDLE = 'IDLE',
  LOADING = 'LOADING',
  SUCCESS = 'SUCCESS',
  ERROR = 'ERROR'
}

export interface ImageFile {
  file: File;
  previewUrl: string;
  base64: string; // Clean base64 without prefix
  mimeType: string;
}

export interface GeneratedImage {
  base64: string;
  mimeType: string;
}

export type PresetPrompt = {
  id: string;
  label: string;
  prompt: string;
  icon: string;
};