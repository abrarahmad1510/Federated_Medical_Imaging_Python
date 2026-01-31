import { createSlice } from '@reduxjs/toolkit';

interface ClientState {
  clients: any[];
  currentClient: any | null;
  loading: boolean;
}

const initialState: ClientState = {
  clients: [],
  currentClient: null,
  loading: false,
};

const clientSlice = createSlice({
  name: 'clients',
  initialState,
  reducers: {},
});

export default clientSlice.reducer;
