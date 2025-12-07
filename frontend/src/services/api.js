/**
 * API Service Layer
 * Centralizes all backend communication for the corn disease detection app
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

/**
 * Custom error class for API errors
 */
class ApiError extends Error {
    constructor(message, status, data) {
        super(message);
        this.name = 'ApiError';
        this.status = status;
        this.data = data;
    }
}

/**
 * Health check endpoint
 * Verifies API is running and model is loaded
 */
export async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);

        if (!response.ok) {
            throw new ApiError(
                'Health check failed',
                response.status,
                await response.json()
            );
        }

        return await response.json();
    } catch (error) {
        if (error instanceof ApiError) {
            throw error;
        }
        throw new ApiError(
            'Unable to connect to server',
            0,
            { originalError: error.message }
        );
    }
}

/**
 * Predict disease from image
 * @param {File} imageFile - Image file to analyze
 * @returns {Promise<Object>} Prediction results
 */
export async function predictDisease(imageFile) {
    if (!imageFile) {
        throw new ApiError('No image file provided', 400, {});
    }

    const formData = new FormData();
    formData.append('image', imageFile);

    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            throw new ApiError(
                data.error || 'Prediction failed',
                response.status,
                data
            );
        }

        return data;
    } catch (error) {
        if (error instanceof ApiError) {
            throw error;
        }
        throw new ApiError(
            'Network error: Unable to reach server',
            0,
            { originalError: error.message }
        );
    }
}

/**
 * Get API base URL (useful for debugging)
 */
export function getApiUrl() {
    return API_BASE_URL;
}

/**
 * Default export with all API functions
 */
export default {
    checkHealth,
    predictDisease,
    getApiUrl,
};
