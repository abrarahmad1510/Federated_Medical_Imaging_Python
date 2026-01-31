import { createSlice } from '@reduxjs/toolkit';

interface ExperimentState {
  experiments: any[];
  currentExperiment: any | null;
  loading: boolean;
}

const initialState: ExperimentState = {
  experiments: [],
  currentExperiment: null,
  loading: false,
};

const experimentSlice = createSlice({
  name: 'experiments',
  initialState,
  reducers: {},
});

export default experimentSlice.reducer;
