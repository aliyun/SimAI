"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.externalEventsMiddleware = exports.externalEventAction = void 0;
var _toolkit = require("@reduxjs/toolkit");
var _tooltipSelectors = require("./selectors/tooltipSelectors");
var externalEventAction = exports.externalEventAction = (0, _toolkit.createAction)('externalEvent');
var externalEventsMiddleware = exports.externalEventsMiddleware = (0, _toolkit.createListenerMiddleware)();

/*
 * We need a Map keyed by event type because this middleware handles MULTIPLE different event types
 * (click, mouseenter, mouseleave, mousedown, mouseup, contextmenu, dblclick, touchstart, touchmove, touchend)
 * from the same DOM element. Different event types should NOT cancel each other's animation frames.
 * For example, a click event and a mousemove event can happen in quick succession and both should be processed.
 * This is different from mouseMoveMiddleware which only handles one event type and uses a single rafId.
 */
var rafIdMap = new Map();
externalEventsMiddleware.startListening({
  actionCreator: externalEventAction,
  effect: (action, listenerApi) => {
    var {
      handler,
      reactEvent
    } = action.payload;
    if (handler == null) {
      return;
    }
    reactEvent.persist();
    var eventType = reactEvent.type;

    // Cancel any pending animation frame for this event type
    var existingRafId = rafIdMap.get(eventType);
    if (existingRafId !== undefined) {
      cancelAnimationFrame(existingRafId);
    }
    var rafId = requestAnimationFrame(() => {
      try {
        /*
         * Here it is important that we get the latest state inside the animation frame callback,
         * not from the outer scope, because there may have been other actions dispatched
         * between the time the event was fired and the animation frame callback is executed.
         * One of those actions is the one that actually sets the active tooltip state!
         */
        var state = listenerApi.getState();
        var nextState = {
          activeCoordinate: (0, _tooltipSelectors.selectActiveTooltipCoordinate)(state),
          activeDataKey: (0, _tooltipSelectors.selectActiveTooltipDataKey)(state),
          activeIndex: (0, _tooltipSelectors.selectActiveTooltipIndex)(state),
          activeLabel: (0, _tooltipSelectors.selectActiveLabel)(state),
          activeTooltipIndex: (0, _tooltipSelectors.selectActiveTooltipIndex)(state),
          isTooltipActive: (0, _tooltipSelectors.selectIsTooltipActive)(state)
        };
        handler(nextState, reactEvent);
      } finally {
        rafIdMap.delete(eventType);
      }
    });
    rafIdMap.set(eventType, rafId);
  }
});