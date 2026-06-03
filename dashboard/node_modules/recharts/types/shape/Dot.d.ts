import * as React from 'react';
import { PresentationAttributesWithProps, RechartsMouseEventHandler } from '../util/types';
interface DotProps {
    className?: string;
    /**
     * The x-coordinate of center in pixels.
     */
    cx?: number;
    /**
     * The y-coordinate of center in pixels.
     */
    cy?: number;
    /**
     * The radius of dot.
     */
    r?: number | string;
    clipDot?: boolean;
    /**
     * The customized event handler of click in this chart.
     */
    onClick?: RechartsMouseEventHandler<Props>;
}
export type Props = PresentationAttributesWithProps<DotProps, SVGCircleElement> & DotProps;
/**
 * Renders a dot in the chart.
 *
 * This component accepts X and Y coordinates in pixels.
 * If you need to position the rectangle based on your chart's data,
 * consider using the {@link ReferenceDot} component instead.
 *
 * @param props
 * @constructor
 */
export declare const Dot: React.FC<Props>;
export {};
