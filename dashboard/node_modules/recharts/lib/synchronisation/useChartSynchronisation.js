"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.useBrushChartSynchronisation = useBrushChartSynchronisation;
exports.useSynchronisedEventsFromOtherCharts = useSynchronisedEventsFromOtherCharts;
exports.useTooltipChartSynchronisation = useTooltipChartSynchronisation;
var _react = require("react");
var _hooks = require("../state/hooks");
var _rootPropsSelectors = require("../state/selectors/rootPropsSelectors");
var _Events = require("../util/Events");
var _optionsSlice = require("../state/optionsSlice");
var _tooltipSlice = require("../state/tooltipSlice");
var _selectors = require("../state/selectors/selectors");
var _tooltipSelectors = require("../state/selectors/tooltipSelectors");
var _syncSelectors = require("./syncSelectors");
var _chartLayoutContext = require("../context/chartLayoutContext");
var _chartDataSlice = require("../state/chartDataSlice");
var _DataUtils = require("../util/DataUtils");
var _excluded = ["x", "y"];
function ownKeys(e, r) { var t = Object.keys(e); if (Object.getOwnPropertySymbols) { var o = Object.getOwnPropertySymbols(e); r && (o = o.filter(function (r) { return Object.getOwnPropertyDescriptor(e, r).enumerable; })), t.push.apply(t, o); } return t; }
function _objectSpread(e) { for (var r = 1; r < arguments.length; r++) { var t = null != arguments[r] ? arguments[r] : {}; r % 2 ? ownKeys(Object(t), !0).forEach(function (r) { _defineProperty(e, r, t[r]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys(Object(t)).forEach(function (r) { Object.defineProperty(e, r, Object.getOwnPropertyDescriptor(t, r)); }); } return e; }
function _defineProperty(e, r, t) { return (r = _toPropertyKey(r)) in e ? Object.defineProperty(e, r, { value: t, enumerable: !0, configurable: !0, writable: !0 }) : e[r] = t, e; }
function _toPropertyKey(t) { var i = _toPrimitive(t, "string"); return "symbol" == typeof i ? i : i + ""; }
function _toPrimitive(t, r) { if ("object" != typeof t || !t) return t; var e = t[Symbol.toPrimitive]; if (void 0 !== e) { var i = e.call(t, r || "default"); if ("object" != typeof i) return i; throw new TypeError("@@toPrimitive must return a primitive value."); } return ("string" === r ? String : Number)(t); }
function _objectWithoutProperties(e, t) { if (null == e) return {}; var o, r, i = _objectWithoutPropertiesLoose(e, t); if (Object.getOwnPropertySymbols) { var n = Object.getOwnPropertySymbols(e); for (r = 0; r < n.length; r++) o = n[r], -1 === t.indexOf(o) && {}.propertyIsEnumerable.call(e, o) && (i[o] = e[o]); } return i; }
function _objectWithoutPropertiesLoose(r, e) { if (null == r) return {}; var t = {}; for (var n in r) if ({}.hasOwnProperty.call(r, n)) { if (-1 !== e.indexOf(n)) continue; t[n] = r[n]; } return t; }
function useTooltipSyncEventsListener() {
  var mySyncId = (0, _hooks.useAppSelector)(_rootPropsSelectors.selectSyncId);
  var myEventEmitter = (0, _hooks.useAppSelector)(_rootPropsSelectors.selectEventEmitter);
  var dispatch = (0, _hooks.useAppDispatch)();
  var syncMethod = (0, _hooks.useAppSelector)(_rootPropsSelectors.selectSyncMethod);
  var tooltipTicks = (0, _hooks.useAppSelector)(_tooltipSelectors.selectTooltipAxisTicks);
  var layout = (0, _chartLayoutContext.useChartLayout)();
  var viewBox = (0, _chartLayoutContext.useViewBox)();
  var className = (0, _hooks.useAppSelector)(state => state.rootProps.className);
  (0, _react.useEffect)(() => {
    if (mySyncId == null) {
      // This chart is not synchronised with any other chart so we don't need to listen for any events.
      return _DataUtils.noop;
    }
    var listener = (incomingSyncId, action, emitter) => {
      if (myEventEmitter === emitter) {
        // We don't want to dispatch actions that we sent ourselves.
        return;
      }
      if (mySyncId !== incomingSyncId) {
        // This event is not for this chart
        return;
      }
      if (syncMethod === 'index') {
        var _action$payload;
        if (viewBox && action !== null && action !== void 0 && (_action$payload = action.payload) !== null && _action$payload !== void 0 && _action$payload.coordinate && action.payload.sourceViewBox) {
          var _action$payload$coord = action.payload.coordinate,
            {
              x: _x,
              y: _y
            } = _action$payload$coord,
            otherCoordinateProps = _objectWithoutProperties(_action$payload$coord, _excluded);
          var {
            x: sourceX,
            y: sourceY,
            width: sourceWidth,
            height: sourceHeight
          } = action.payload.sourceViewBox;
          var scaledCoordinate = _objectSpread(_objectSpread({}, otherCoordinateProps), {}, {
            x: viewBox.x + (sourceWidth ? (_x - sourceX) / sourceWidth : 0) * viewBox.width,
            y: viewBox.y + (sourceHeight ? (_y - sourceY) / sourceHeight : 0) * viewBox.height
          });
          dispatch(_objectSpread(_objectSpread({}, action), {}, {
            payload: _objectSpread(_objectSpread({}, action.payload), {}, {
              coordinate: scaledCoordinate
            })
          }));
        } else {
          dispatch(action);
        }
        return;
      }
      if (tooltipTicks == null) {
        // for the other two sync methods, we need the ticks to be available
        return;
      }
      var activeTick;
      if (typeof syncMethod === 'function') {
        /*
         * This is what the data shape in 2.x CategoricalChartState used to look like.
         * In 3.x we store things differently but let's try to keep the old shape for compatibility.
         */
        var syncMethodParam = {
          activeTooltipIndex: action.payload.index == null ? undefined : Number(action.payload.index),
          isTooltipActive: action.payload.active,
          activeIndex: action.payload.index == null ? undefined : Number(action.payload.index),
          activeLabel: action.payload.label,
          activeDataKey: action.payload.dataKey,
          activeCoordinate: action.payload.coordinate
        };
        // Call a callback function. If there is an application specific algorithm
        var activeTooltipIndex = syncMethod(tooltipTicks, syncMethodParam);
        activeTick = tooltipTicks[activeTooltipIndex];
      } else if (syncMethod === 'value') {
        // labels are always strings, tick.value might be a string or a number, depending on axis type
        activeTick = tooltipTicks.find(tick => String(tick.value) === action.payload.label);
      }
      var {
        coordinate
      } = action.payload;
      if (activeTick == null || action.payload.active === false || coordinate == null || viewBox == null) {
        dispatch((0, _tooltipSlice.setSyncInteraction)({
          active: false,
          coordinate: undefined,
          dataKey: undefined,
          index: null,
          label: undefined,
          sourceViewBox: undefined,
          graphicalItemId: undefined
        }));
        return;
      }
      var {
        x,
        y
      } = coordinate;
      var validateChartX = Math.min(x, viewBox.x + viewBox.width);
      var validateChartY = Math.min(y, viewBox.y + viewBox.height);
      var activeCoordinate = {
        x: layout === 'horizontal' ? activeTick.coordinate : validateChartX,
        y: layout === 'horizontal' ? validateChartY : activeTick.coordinate
      };
      var syncAction = (0, _tooltipSlice.setSyncInteraction)({
        active: action.payload.active,
        coordinate: activeCoordinate,
        dataKey: action.payload.dataKey,
        index: String(activeTick.index),
        label: action.payload.label,
        sourceViewBox: action.payload.sourceViewBox,
        graphicalItemId: action.payload.graphicalItemId
      });
      dispatch(syncAction);
    };
    _Events.eventCenter.on(_Events.TOOLTIP_SYNC_EVENT, listener);
    return () => {
      _Events.eventCenter.off(_Events.TOOLTIP_SYNC_EVENT, listener);
    };
  }, [className, dispatch, myEventEmitter, mySyncId, syncMethod, tooltipTicks, layout, viewBox]);
}
function useBrushSyncEventsListener() {
  var mySyncId = (0, _hooks.useAppSelector)(_rootPropsSelectors.selectSyncId);
  var myEventEmitter = (0, _hooks.useAppSelector)(_rootPropsSelectors.selectEventEmitter);
  var dispatch = (0, _hooks.useAppDispatch)();
  (0, _react.useEffect)(() => {
    if (mySyncId == null) {
      // This chart is not synchronised with any other chart so we don't need to listen for any events.
      return _DataUtils.noop;
    }
    var listener = (incomingSyncId, action, emitter) => {
      if (myEventEmitter === emitter) {
        // We don't want to dispatch actions that we sent ourselves.
        return;
      }
      if (mySyncId === incomingSyncId) {
        dispatch((0, _chartDataSlice.setDataStartEndIndexes)(action));
      }
    };
    _Events.eventCenter.on(_Events.BRUSH_SYNC_EVENT, listener);
    return () => {
      _Events.eventCenter.off(_Events.BRUSH_SYNC_EVENT, listener);
    };
  }, [dispatch, myEventEmitter, mySyncId]);
}

/**
 * Will receive synchronisation events from other charts.
 *
 * Reads syncMethod from state and decides how to synchronise the tooltip based on that.
 *
 * @returns void
 */
function useSynchronisedEventsFromOtherCharts() {
  var dispatch = (0, _hooks.useAppDispatch)();
  (0, _react.useEffect)(() => {
    dispatch((0, _optionsSlice.createEventEmitter)());
  }, [dispatch]);
  useTooltipSyncEventsListener();
  useBrushSyncEventsListener();
}

/**
 * Will send events to other charts.
 * If syncId is undefined, no events will be sent.
 *
 * This ignores the syncMethod, because that is set and computed on the receiving end.
 *
 * @param tooltipEventType from Tooltip
 * @param trigger from Tooltip
 * @param activeCoordinate from state
 * @param activeLabel from state
 * @param activeIndex from state
 * @param isTooltipActive from state
 * @returns void
 */
function useTooltipChartSynchronisation(tooltipEventType, trigger, activeCoordinate, activeLabel, activeIndex, isTooltipActive) {
  var activeDataKey = (0, _hooks.useAppSelector)(state => (0, _selectors.selectTooltipDataKey)(state, tooltipEventType, trigger));
  var eventEmitterSymbol = (0, _hooks.useAppSelector)(_rootPropsSelectors.selectEventEmitter);
  var syncId = (0, _hooks.useAppSelector)(_rootPropsSelectors.selectSyncId);
  var syncMethod = (0, _hooks.useAppSelector)(_rootPropsSelectors.selectSyncMethod);
  var tooltipState = (0, _hooks.useAppSelector)(_syncSelectors.selectSynchronisedTooltipState);
  var isReceivingSynchronisation = tooltipState === null || tooltipState === void 0 ? void 0 : tooltipState.active;
  var viewBox = (0, _chartLayoutContext.useViewBox)();
  (0, _react.useEffect)(() => {
    if (isReceivingSynchronisation) {
      /*
       * This chart currently has active tooltip, synchronised from another chart.
       * Let's not send any outgoing synchronisation events while that's happening
       * to avoid infinite loops.
       */
      return;
    }
    if (syncId == null) {
      /*
       * syncId is not set, means that this chart is not synchronised with any other chart,
       * means we don't need to send synchronisation events
       */
      return;
    }
    if (eventEmitterSymbol == null) {
      /*
       * When using Recharts internal hooks and selectors outside charts context,
       * these properties will be undefined. Let's return silently instead of throwing an error.
       */
      return;
    }
    var syncAction = (0, _tooltipSlice.setSyncInteraction)({
      active: isTooltipActive,
      coordinate: activeCoordinate,
      dataKey: activeDataKey,
      index: activeIndex,
      label: typeof activeLabel === 'number' ? String(activeLabel) : activeLabel,
      sourceViewBox: viewBox,
      graphicalItemId: undefined
    });
    _Events.eventCenter.emit(_Events.TOOLTIP_SYNC_EVENT, syncId, syncAction, eventEmitterSymbol);
  }, [isReceivingSynchronisation, activeCoordinate, activeDataKey, activeIndex, activeLabel, eventEmitterSymbol, syncId, syncMethod, isTooltipActive, viewBox]);
}
function useBrushChartSynchronisation() {
  var syncId = (0, _hooks.useAppSelector)(_rootPropsSelectors.selectSyncId);
  var eventEmitterSymbol = (0, _hooks.useAppSelector)(_rootPropsSelectors.selectEventEmitter);
  var brushStartIndex = (0, _hooks.useAppSelector)(state => state.chartData.dataStartIndex);
  var brushEndIndex = (0, _hooks.useAppSelector)(state => state.chartData.dataEndIndex);
  (0, _react.useEffect)(() => {
    if (syncId == null || brushStartIndex == null || brushEndIndex == null || eventEmitterSymbol == null) {
      return;
    }
    var syncAction = {
      startIndex: brushStartIndex,
      endIndex: brushEndIndex
    };
    _Events.eventCenter.emit(_Events.BRUSH_SYNC_EVENT, syncId, syncAction, eventEmitterSymbol);
  }, [brushEndIndex, brushStartIndex, eventEmitterSymbol, syncId]);
}