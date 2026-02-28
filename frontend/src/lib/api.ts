const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface User {
  id: number;
  username: string;
  email: string;
}

export interface LoginData {
  username: string;
  password: string;
}

export interface RegisterData {
  username: string;
  email: string;
  password: string;
}

export interface ForecastItem {
  id: number;
  name: string | null;
  input_images_count: number;
  processing_time: number;
  created_at: string;
}

export interface ForecastDetail extends ForecastItem {
  generated_image: string;
}

export interface ApiResponse<T> {
  success: boolean;
  message: string;
  data?: T;
}

class ApiService {
  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;
    
    const config: RequestInit = {
      credentials: 'include', // Include cookies for session management
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  // Health check
  async checkHealth(): Promise<{ status: string; model_loaded: boolean }> {
    return this.request('/health');
  }

  // Authentication
  async login(data: LoginData): Promise<{ success: boolean; user: User; message: string }> {
    return this.request('/auth/login', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async register(data: RegisterData): Promise<{ success: boolean; user: User; message: string }> {
    return this.request('/auth/register', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async logout(): Promise<{ success: boolean; message: string }> {
    return this.request('/auth/logout', {
      method: 'POST',
    });
  }

  async getCurrentUser(): Promise<{ success: boolean; user: User; message: string }> {
    return this.request('/auth/me');
  }

  // Forecasts
  async generateForecast(
    files: File[],
    forecastName: string = 'Untitled Forecast'
  ): Promise<{
    success: boolean;
    generated_image: string;
    processing_time: number;
    message: string;
    forecast_id: number;
  }> {
    const formData = new FormData();
    
    files.forEach((file) => {
      formData.append('files', file);
    });
    formData.append('forecast_name', forecastName);

    return this.request('/predict', {
      method: 'POST',
      headers: {}, // Remove Content-Type to let browser set it for FormData
      body: formData,
    });
  }

  async getForecasts(): Promise<{
    success: boolean;
    forecasts: ForecastItem[];
    message: string;
  }> {
    return this.request('/forecasts');
  }

  async getForecast(id: number): Promise<{
    success: boolean;
    forecast: ForecastDetail;
    message: string;
  }> {
    return this.request(`/forecasts/${id}`);
  }
}

export const api = new ApiService();

// Auth context helpers
export const isAuthenticated = async (): Promise<boolean> => {
  try {
    await api.getCurrentUser();
    return true;
  } catch {
    return false;
  }
};