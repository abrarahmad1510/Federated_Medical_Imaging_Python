import { createSlice } from '@reduxjs/toolkit';

interface ModelState {
  models: any[];
  currentModel: any | null;
  loading: boolean;
}

const initialState: ModelState = {
  models: [],
  currentModel: null,
  loading: false,
};

const modelSlice = createSlice({
  name: 'models',
  initialState,
  reducers: {},
});

export default modelSlice.reducer;
