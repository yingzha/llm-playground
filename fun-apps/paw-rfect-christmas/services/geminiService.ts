import { GoogleGenAI } from "@google/genai";
import { GeneratedImage } from "../types";

/**
 * Edits a pet image to have a Christmas theme based on user prompt.
 */
export const generateChristmasEdit = async (
  imageBase64: string,
  imageMimeType: string,
  userPrompt: string
): Promise<GeneratedImage> => {
  try {
    // Initialize inside the function to ensure we use the most up-to-date API Key
    const ai = new GoogleGenAI({ apiKey: process.env.GOOGLE_API_KEY });
    
    // --- Step 1: Verify the image contains a pet ---
    // We use a fast model to check the image content first to avoid wasting generation resources
    // on non-pet images.
    try {
      const checkResponse = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: {
          parts: [
            {
              text: `Analyze this image. Does it contain a visible real animal or pet (like a dog, cat, bird, hamster, rabbit, etc)? 
              If it is a human, a landscape, or an object without an animal, return false.
              Return a JSON object with a single property "hasPet" (boolean).`
            },
            {
              inlineData: {
                mimeType: imageMimeType,
                data: imageBase64
              }
            }
          ]
        },
        config: {
          responseMimeType: 'application/json'
        }
      });

      const checkText = checkResponse.text;
      if (checkText) {
        const checkResult = JSON.parse(checkText);
        // Explicitly check for false to avoid false positives on malformed JSON
        if (checkResult.hasPet === false) {
           throw new Error("We couldn't detect a pet in this photo. Please upload a clear photo of your furry friend!");
        }
      }
    } catch (checkError: any) {
      // If the specific "no pet" error was thrown, re-throw it.
      if (checkError.message.includes("couldn't detect a pet")) {
        throw checkError;
      }
      // If the check fails for technical reasons (e.g. model overload), we log it but might choose to proceed 
      // or fail safe. Here we'll fail safe to ensure quality.
      console.warn("Pet detection check failed or encountered an error:", checkError);
    }

    // --- Step 2: Generate the Christmas Edit ---
    const model = 'gemini-3-pro-image-preview';
    
    // Construct a strong prompt for the model with emphasis on subject preservation
    const fullPrompt = `Edit this photo to create a Christmas-themed masterpiece. 
    CRITICAL: You MUST preserve the exact appearance, breed, fur color/pattern, and pose of the pet in the original image. Do not generate a different animal.
    
    Task:
    1. Keep the pet exactly as is (identity preservation is priority #1).
    2. Change the background or add elements to create a festive Christmas atmosphere.
    3. Ensure the lighting on the pet matches the new scene (warm, magical holiday lighting).
    
    User Instructions: ${userPrompt || "Decorate the surroundings with Christmas lights, snow, and gifts."}
    
    Style: Photorealistic, cinematic lighting, 8k resolution, magical holiday vibes.`;

    const response = await ai.models.generateContent({
      model: model,
      contents: {
        parts: [
          {
            text: fullPrompt
          },
          {
            inlineData: {
              mimeType: imageMimeType,
              data: imageBase64
            }
          }
        ]
      },
      config: {
        // gemini-3-pro-image-preview supports imageConfig
        imageConfig: {
          imageSize: '1K', // Default to 1K for good quality
        }
      }
    });

    // Parse the response to find the image
    const parts = response.candidates?.[0]?.content?.parts;
    
    if (!parts) {
      throw new Error("No content generated.");
    }

    // Look for the part with inlineData (the image)
    const imagePart = parts.find(p => p.inlineData);

    if (imagePart && imagePart.inlineData) {
      return {
        base64: imagePart.inlineData.data,
        mimeType: imagePart.inlineData.mimeType || 'image/png'
      };
    }

    throw new Error("The model did not return an image. It might have refused the request.");
  } catch (error: any) {
    console.error("Gemini API Error:", error);
    
    // Check for common permission errors to give better feedback
    if (error.message?.includes('403') || error.message?.includes('404')) {
        throw new Error("Access denied. Please ensure you have selected a valid API key with billing enabled for Gemini 3 Pro.");
    }
    
    throw error;
  }
};