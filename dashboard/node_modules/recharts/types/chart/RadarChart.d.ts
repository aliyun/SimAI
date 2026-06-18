import * as React from 'react';
import { PolarChartProps, PolarLayout } from '../util/types';
export declare const defaultRadarChartProps: {
    readonly layout: "centric";
    readonly startAngle: 90;
    readonly endAngle: -270;
    readonly accessibilityLayer: true;
    readonly stackOffset: "none";
    readonly barCategoryGap: "10%";
    readonly barGap: 4;
    readonly margin: import("../util/types").Margin;
    readonly reverseStackOrder: false;
    readonly syncMethod: "index";
    readonly responsive: false;
    readonly cx: "50%";
    readonly cy: "50%";
    readonly innerRadius: 0;
    readonly outerRadius: "80%";
};
/**
 * @consumes ResponsiveContainerContext
 * @provides PolarViewBoxContext
 * @provides PolarChartContext
 */
export declare const RadarChart: React.ForwardRefExoticComponent<Omit<PolarChartProps, "layout" | "startAngle" | "endAngle"> & {
    /**
     * The layout of chart defines the orientation of axes, graphical items, and tooltip.
     *
     * @defaultValue centric
     */
    layout?: PolarLayout;
    /**
     * Angle in degrees from which the chart should start.
     * @defaultValue 90
     *
     */
    startAngle?: number;
    /**
     * Angle, in degrees, at which the chart should end.
     *
     * @defaultValue -270
     */
    endAngle?: number;
} & React.RefAttributes<SVGSVGElement>>;
