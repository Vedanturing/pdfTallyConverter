import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import axios from 'axios';
import { API_URL } from '../config';

export interface User {
  id: string;
  name: string;
  email: string;
  created_at: string;
  is_active: boolean;
}

export interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;

  // Actions
  login: (email: string, password: string) => Promise<boolean>;
  signup: (name: string, email: string, password: string) => Promise<boolean>;
  logout: () => void;
  clearError: () => void;
  checkAuth: () => Promise<void>;
  deleteAccount: () => Promise<boolean>;
  updateProfile: (name?: string, email?: string) => Promise<boolean>;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,

      login: async (email: string, password: string) => {
        set({ isLoading: true, error: null });
        
        try {
          const response = await axios.post(`${API_URL}/auth/login`, {
            email,
            password
          });

          const { user, access_token } = response.data;
          
          // Set token in axios defaults for future requests
          axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
          
          set({
            user,
            token: access_token,
            isAuthenticated: true,
            isLoading: false,
            error: null
          });

          return true;
        } catch (error: any) {
          const errorMessage = error.response?.data?.detail || 'Login failed';
          set({
            isLoading: false,
            error: errorMessage
          });
          return false;
        }
      },

      signup: async (name: string, email: string, password: string) => {
        set({ isLoading: true, error: null });
        
        try {
          const response = await axios.post(`${API_URL}/auth/signup`, {
            name,
            email,
            password
          });

          const { user, access_token } = response.data;
          
          // Set token in axios defaults for future requests
          axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
          
          set({
            user,
            token: access_token,
            isAuthenticated: true,
            isLoading: false,
            error: null
          });

          return true;
        } catch (error: any) {
          const errorMessage = error.response?.data?.detail || 'Signup failed';
          set({
            isLoading: false,
            error: errorMessage
          });
          return false;
        }
      },

      logout: () => {
        // Remove token from axios defaults
        delete axios.defaults.headers.common['Authorization'];
        
        set({
          user: null,
          token: null,
          isAuthenticated: false,
          error: null
        });
      },

      clearError: () => {
        set({ error: null });
      },

      checkAuth: async () => {
        const { token } = get();
        
        if (!token) {
          return;
        }

        try {
          // Set token in axios defaults
          axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
          
          const response = await axios.get(`${API_URL}/auth/me`);
          const user = response.data;

          set({
            user,
            isAuthenticated: true
          });
        } catch (error) {
          // Token is invalid, clear auth state
          get().logout();
        }
      },

      deleteAccount: async () => {
        set({ isLoading: true, error: null });
        
        try {
          await axios.delete(`${API_URL}/auth/delete_account`);
          
          // Clear auth state after successful deletion
          get().logout();
          
          set({ isLoading: false });
          return true;
        } catch (error: any) {
          const errorMessage = error.response?.data?.detail || 'Account deletion failed';
          set({
            isLoading: false,
            error: errorMessage
          });
          return false;
        }
      },

      updateProfile: async (name?: string, email?: string) => {
        set({ isLoading: true, error: null });
        
        try {
          const updateData: any = {};
          if (name !== undefined) updateData.name = name;
          if (email !== undefined) updateData.email = email;

          const response = await axios.put(`${API_URL}/auth/me`, updateData);
          const updatedUser = response.data;

          set({
            user: updatedUser,
            isLoading: false,
            error: null
          });

          return true;
        } catch (error: any) {
          const errorMessage = error.response?.data?.detail || 'Profile update failed';
          set({
            isLoading: false,
            error: errorMessage
          });
          return false;
        }
      }
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({
        user: state.user,
        token: state.token,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
); 