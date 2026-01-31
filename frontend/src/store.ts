import { configureStore } from '@reduxjs/toolkit';
import authReducer from './features/auth/authSlice';
import experimentReducer from './features/experiments/experimentSlice';
import modelReducer from './features/models/modelSlice';
import clientReducer from './features/clients/clientSlice';

export const store = configureStore({
  reducer: {
    auth: authReducer,
    experiments: experimentReducer,
    models: modelReducer,
    clients: clientReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['persist/PERSIST'],
      },
    }),
  devTools: process.env.NODE_ENV !== 'production',
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
