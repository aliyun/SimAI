"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.getRadialCursorPoints = getRadialCursorPoints;
var _PolarUtils = require("../PolarUtils");
/**
 * Only applicable for radial layouts
 * @param {Object} activeCoordinate ChartCoordinate
 * @returns {Object} RadialCursorPoints
 */
function getRadialCursorPoints(activeCoordinate) {
  var {
    cx,
    cy,
    radius,
    startAngle,
    endAngle
  } = activeCoordinate;
  var startPoint = (0, _PolarUtils.polarToCartesian)(cx, cy, radius, startAngle);
  var endPoint = (0, _PolarUtils.polarToCartesian)(cx, cy, radius, endAngle);
  return {
    points: [startPoint, endPoint],
    cx,
    cy,
    radius,
    startAngle,
    endAngle
  };
}