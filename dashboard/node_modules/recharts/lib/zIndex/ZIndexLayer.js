"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.ZIndexLayer = ZIndexLayer;
var _react = require("react");
var _reactDom = require("react-dom");
var _DataUtils = require("../util/DataUtils");
var _hooks = require("../state/hooks");
var _zIndexSelectors = require("./zIndexSelectors");
var _zIndexSlice = require("../state/zIndexSlice");
var _chartLayoutContext = require("../context/chartLayoutContext");
var _PanoramaContext = require("../context/PanoramaContext");
/**
 * @since 3.4
 */

/**
 * A layer that renders its children into a portal corresponding to the given zIndex.
 * We can't use regular CSS `z-index` because SVG does not support it.
 * So instead, we create separate DOM nodes for each zIndex layer
 * and render the children into the corresponding DOM node using React portals.
 *
 * This component must be used inside a Chart component.
 *
 * @param zIndex numeric zIndex value, higher values are rendered on top of lower values
 * @param children the content to render inside this zIndex layer
 *
 * @since 3.4
 */
function ZIndexLayer(_ref) {
  var {
    zIndex,
    children
  } = _ref;
  /*
   * If we are outside of chart, then we can't rely on the zIndex portal state,
   * so we just render normally.
   */
  var isInChartContext = (0, _chartLayoutContext.useIsInChartContext)();
  /*
   * If zIndex is undefined then we render normally without portals.
   * Also, if zIndex is 0, we render normally without portals,
   * because 0 is the default layer that does not need a portal.
   */
  var shouldRenderInPortal = isInChartContext && zIndex !== undefined && zIndex !== 0;
  var isPanorama = (0, _PanoramaContext.useIsPanorama)();
  var dispatch = (0, _hooks.useAppDispatch)();
  (0, _react.useLayoutEffect)(() => {
    if (!shouldRenderInPortal) {
      // Nothing to do. We have to call the hook because of the rules of hooks.
      return _DataUtils.noop;
    }
    /*
     * Because zIndexes are dynamic (meaning, we're not working with a predefined set of layers,
     * but we allow users to define any zIndex at any time), we need to register
     * the requested zIndex in the global store. This way, the ZIndexPortals component
     * can render the corresponding portals and only the requested ones.
     */
    dispatch((0, _zIndexSlice.registerZIndexPortal)({
      zIndex
    }));
    return () => {
      dispatch((0, _zIndexSlice.unregisterZIndexPortal)({
        zIndex
      }));
    };
  }, [dispatch, zIndex, shouldRenderInPortal]);
  var portalElement = (0, _hooks.useAppSelector)(state => (0, _zIndexSelectors.selectZIndexPortalElement)(state, zIndex, isPanorama));
  if (!shouldRenderInPortal) {
    // If no zIndex is provided or zIndex is 0, render normally without portals
    return children;
  }
  if (!portalElement) {
    /*
     * If we don't have a portal element yet, this means that the registration
     * has not been processed yet by the ZIndexPortals component.
     * So here we render null and wait for the next render cycle.
     */
    return null;
  }
  return /*#__PURE__*/(0, _reactDom.createPortal)(children, portalElement);
}