import { createSlice } from '@reduxjs/toolkit';
var initialState = null;
var reducers = {
  updatePolarOptions: (_state, action) => {
    return action.payload;
  }
};
var polarOptionsSlice = createSlice({
  name: 'polarOptions',
  initialState,
  reducers
});
export var {
  updatePolarOptions
} = polarOptionsSlice.actions;
export var polarOptionsReducer = polarOptionsSlice.reducer;