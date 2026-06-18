"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.updatePolarOptions = exports.polarOptionsReducer = void 0;
var _toolkit = require("@reduxjs/toolkit");
var initialState = null;
var reducers = {
  updatePolarOptions: (_state, action) => {
    return action.payload;
  }
};
var polarOptionsSlice = (0, _toolkit.createSlice)({
  name: 'polarOptions',
  initialState,
  reducers
});
var {
  updatePolarOptions
} = polarOptionsSlice.actions;
exports.updatePolarOptions = updatePolarOptions;
var polarOptionsReducer = exports.polarOptionsReducer = polarOptionsSlice.reducer;