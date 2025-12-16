import { PresetPrompt } from './types';

export const PRESET_STYLES: PresetPrompt[] = [
  {
    id: 'cozy',
    label: 'Cozy Fireplace',
    prompt: 'A warm, cozy Christmas scene with a fireplace in the background, soft lighting, and stockings.',
    icon: '🔥'
  },
  {
    id: 'snowy',
    label: 'Winter Wonderland',
    prompt: 'A magical outdoor snowy scene with falling snowflakes, pine trees, and winter accessories.',
    icon: '❄️'
  },
  {
    id: 'santa',
    label: 'Santa Helper',
    prompt: 'The pet wearing a cute Santa hat and red scarf, surrounded by wrapped gifts.',
    icon: '🎅'
  },
  {
    id: 'lights',
    label: 'Festive Lights',
    prompt: 'Surrounded by glowing colorful Christmas lights with a bokeh effect, very festive and bright.',
    icon: '💡'
  }
];

export const SYSTEM_INSTRUCTION = `You are a professional image editor specializing in pet photography and Christmas themes. 
Your goal is to transform user-uploaded pet photos into festive Christmas masterpieces.
Ensure the pet remains the central focus and is recognizable, but enhance the environment, lighting, and accessories to fit the holiday spirit.
High quality, photorealistic, and magical.`;
