"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.updateYAxisWidth = exports.replaceZAxis = exports.replaceYAxis = exports.replaceXAxis = exports.removeZAxis = exports.removeYAxis = exports.removeXAxis = exports.defaultAxisId = exports.cartesianAxisReducer = exports.addZAxis = exports.addYAxis = exports.addXAxis = void 0;
var _toolkit = require("@reduxjs/toolkit");
var _immer = require("immer");
function ownKeys(e, r) { var t = Object.keys(e); if (Object.getOwnPropertySymbols) { var o = Object.getOwnPropertySymbols(e); r && (o = o.filter(function (r) { return Object.getOwnPropertyDescriptor(e, r).enumerable; })), t.push.apply(t, o); } return t; }
function _objectSpread(e) { for (var r = 1; r < arguments.length; r++) { var t = null != arguments[r] ? arguments[r] : {}; r % 2 ? ownKeys(Object(t), !0).forEach(function (r) { _defineProperty(e, r, t[r]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys(Object(t)).forEach(function (r) { Object.defineProperty(e, r, Object.getOwnPropertyDescriptor(t, r)); }); } return e; }
function _defineProperty(e, r, t) { return (r = _toPropertyKey(r)) in e ? Object.defineProperty(e, r, { value: t, enumerable: !0, configurable: !0, writable: !0 }) : e[r] = t, e; }
function _toPropertyKey(t) { var i = _toPrimitive(t, "string"); return "symbol" == typeof i ? i : i + ""; }
function _toPrimitive(t, r) { if ("object" != typeof t || !t) return t; var e = t[Symbol.toPrimitive]; if (void 0 !== e) { var i = e.call(t, r || "default"); if ("object" != typeof i) return i; throw new TypeError("@@toPrimitive must return a primitive value."); } return ("string" === r ? String : Number)(t); }
/**
 * @inline
 */

var defaultAxisId = exports.defaultAxisId = 0;

/**
 * Properties shared in X, Y, and Z axes.
 * User defined axis settings, coming from props.
 */

/**
 * These are the external props, visible for users as they set them using our public API.
 * There is all sorts of internal computed things based on these, but they will come through selectors.
 *
 * Properties shared between X and Y axes
 */

/**
 * Z axis is special because it's never displayed. It controls the size of Scatter dots,
 * but it never displays ticks anywhere.
 */

var initialState = {
  xAxis: {},
  yAxis: {},
  zAxis: {}
};

/**
 * This is the slice where each individual Axis element pushes its own configuration.
 * Prefer to use this one instead of axisSlice.
 */
var cartesianAxisSlice = (0, _toolkit.createSlice)({
  name: 'cartesianAxis',
  initialState,
  reducers: {
    addXAxis: {
      reducer(state, action) {
        state.xAxis[action.payload.id] = (0, _immer.castDraft)(action.payload);
      },
      prepare: (0, _toolkit.prepareAutoBatched)()
    },
    replaceXAxis: {
      reducer(state, action) {
        var {
          prev,
          next
        } = action.payload;
        if (state.xAxis[prev.id] !== undefined) {
          if (prev.id !== next.id) {
            delete state.xAxis[prev.id];
          }
          state.xAxis[next.id] = (0, _immer.castDraft)(next);
        }
      },
      prepare: (0, _toolkit.prepareAutoBatched)()
    },
    removeXAxis: {
      reducer(state, action) {
        delete state.xAxis[action.payload.id];
      },
      prepare: (0, _toolkit.prepareAutoBatched)()
    },
    addYAxis: {
      reducer(state, action) {
        state.yAxis[action.payload.id] = (0, _immer.castDraft)(action.payload);
      },
      prepare: (0, _toolkit.prepareAutoBatched)()
    },
    replaceYAxis: {
      reducer(state, action) {
        var {
          prev,
          next
        } = action.payload;
        if (state.yAxis[prev.id] !== undefined) {
          if (prev.id !== next.id) {
            delete state.yAxis[prev.id];
          }
          state.yAxis[next.id] = (0, _immer.castDraft)(next);
        }
      },
      prepare: (0, _toolkit.prepareAutoBatched)()
    },
    removeYAxis: {
      reducer(state, action) {
        delete state.yAxis[action.payload.id];
      },
      prepare: (0, _toolkit.prepareAutoBatched)()
    },
    addZAxis: {
      reducer(state, action) {
        state.zAxis[action.payload.id] = (0, _immer.castDraft)(action.payload);
      },
      prepare: (0, _toolkit.prepareAutoBatched)()
    },
    replaceZAxis: {
      reducer(state, action) {
        var {
          prev,
          next
        } = action.payload;
        if (state.zAxis[prev.id] !== undefined) {
          if (prev.id !== next.id) {
            delete state.zAxis[prev.id];
          }
          state.zAxis[next.id] = (0, _immer.castDraft)(next);
        }
      },
      prepare: (0, _toolkit.prepareAutoBatched)()
    },
    removeZAxis: {
      reducer(state, action) {
        delete state.zAxis[action.payload.id];
      },
      prepare: (0, _toolkit.prepareAutoBatched)()
    },
    updateYAxisWidth(state, action) {
      var {
        id,
        width
      } = action.payload;
      var axis = state.yAxis[id];
      if (axis) {
        var _history$;
        var history = axis.widthHistory || [];
        // An oscillation is detected when the new width is the same as the width before the last one.
        // This is a simple A -> B -> A pattern. If the next width is B, and the difference is less than 1 pixel, we ignore it.
        if (history.length === 3 && history[0] === history[2] && width === history[1] && width !== axis.width && Math.abs(width - ((_history$ = history[0]) !== null && _history$ !== void 0 ? _history$ : 0)) <= 1) {
          return;
        }
        var newHistory = [...history, width].slice(-3);
        state.yAxis[id] = _objectSpread(_objectSpread({}, axis), {}, {
          width,
          widthHistory: newHistory
        });
      }
    }
  }
});
var {
  addXAxis,
  replaceXAxis,
  removeXAxis,
  addYAxis,
  replaceYAxis,
  removeYAxis,
  addZAxis,
  replaceZAxis,
  removeZAxis,
  updateYAxisWidth
} = cartesianAxisSlice.actions;
exports.updateYAxisWidth = updateYAxisWidth;
exports.removeZAxis = removeZAxis;
exports.replaceZAxis = replaceZAxis;
exports.addZAxis = addZAxis;
exports.removeYAxis = removeYAxis;
exports.replaceYAxis = replaceYAxis;
exports.addYAxis = addYAxis;
exports.removeXAxis = removeXAxis;
exports.replaceXAxis = replaceXAxis;
exports.addXAxis = addXAxis;
var cartesianAxisReducer = exports.cartesianAxisReducer = cartesianAxisSlice.reducer;