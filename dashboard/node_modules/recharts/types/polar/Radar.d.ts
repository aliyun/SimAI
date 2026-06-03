import * as React from 'react';
import { MouseEvent, ReactElement, SVGProps } from 'react';
import { ImplicitLabelListType } from '../component/LabelList';
import { ActiveDotType, AnimationDuration, AnimationTiming, DataKey, DotType, LegendType, TooltipType } from '../util/types';
import { ZIndexable } from '../zIndex/ZIndexLayer';
import { RechartsScale } from '../util/scale/RechartsScale';
interface RadarPoint {
    x: number;
    y: number;
    cx?: number;
    cy?: number;
    angle: number;
    radius?: number;
    value?: number;
    payload?: any;
    name?: string;
}
interface RadarProps extends ZIndexable {
    /**
     * @defaultValue true
     */
    activeDot?: ActiveDotType;
    /**
     * @defaultValue 0
     */
    angleAxisId?: string | number;
    /**
     * Specifies when the animation should begin, the unit of this option is ms.
     * @defaultValue 0
     */
    animationBegin?: number;
    /**
     * Specifies the duration of animation, the unit of this option is ms.
     * @defaultValue 1500
     */
    animationDuration?: AnimationDuration;
    /**
     * The type of easing function.
     * @defaultValue ease
     */
    animationEasing?: AnimationTiming;
    baseLinePoints?: RadarPoint[];
    className?: string;
    connectNulls?: boolean;
    /**
     * Decides how to extract the value of this Radar from the data:
     * - `string`: the name of the field in the data object;
     * - `number`: the index of the field in the data;
     * - `function`: a function that receives the data object and returns the value of this Radar.
     */
    dataKey?: DataKey<any>;
    /**
     * Renders a circle element at each data point. Options:
     *
     * - `false`: no dots are drawn;
     * - `true`: renders the dots with default settings;
     * - `object`: the props of the dot. This will be merged with the internal calculated props of each dot;
     * - `ReactElement`: the custom dot element;
     * - `function`: a render function of the custom dot.
     *
     * @defaultValue false
     */
    dot?: DotType;
    /**
     * @defaultValue false
     */
    hide?: boolean;
    /**
     * If set false, animation of polygon will be disabled.
     * If set "auto", the animation will be disabled in SSR and enabled in browser.
     * @defaultValue auto
     */
    isAnimationActive?: boolean | 'auto';
    isRange?: boolean;
    /**
     * Renders one label for each point. Options:
     * - `true`: renders default labels;
     * - `false`: no labels are rendered;
     * - `object`: the props of LabelList component;
     * - `ReactElement`: a custom label element;
     * - `function`: a render function of custom label.
     *
     * @defaultValue false
     */
    label?: ImplicitLabelListType;
    /**
     * The type of icon in legend.  If set to 'none', no legend item will be rendered.
     * @defaultValue rect
     */
    legendType?: LegendType;
    /**
     * The customized event handler of animation end
     */
    onAnimationEnd?: () => void;
    /**
     * The customized event handler of animation start
     */
    onAnimationStart?: () => void;
    onMouseEnter?: (props: any, e: MouseEvent<SVGPolygonElement>) => void;
    onMouseLeave?: (props: any, e: MouseEvent<SVGPolygonElement>) => void;
    /**
     * The coordinates of all the vertexes of the radar shape, like an array of objects with x and y coordinates.
     */
    points?: RadarPoint[];
    /**
     * @defaultValue 0
     */
    radiusAxisId?: string | number;
    /**
     * If set a ReactElement, the shape of radar can be customized.
     * If set a function, the function will be called to render customized shape.
     */
    shape?: ReactElement<SVGElement> | ((props: any) => ReactElement<SVGElement>);
    tooltipType?: TooltipType;
    /**
     * @defaultValue 100
     */
    zIndex?: number;
}
export type RadiusAxisForRadar = {
    scale: RechartsScale;
};
export type AngleAxisForRadar = {
    scale: RechartsScale;
    type: 'number' | 'category';
    dataKey: DataKey<any> | undefined;
    cx: number;
    cy: number;
};
export type Props = Omit<SVGProps<SVGElement>, 'onMouseEnter' | 'onMouseLeave' | 'points' | 'ref'> & RadarProps;
export type RadarComposedData = {
    points: RadarPoint[];
    baseLinePoints: RadarPoint[];
    isRange: boolean;
};
export declare function computeRadarPoints({ radiusAxis, angleAxis, displayedData, dataKey, bandSize, }: {
    radiusAxis: RadiusAxisForRadar;
    angleAxis: AngleAxisForRadar;
    displayedData: any[];
    dataKey: RadarProps['dataKey'];
    bandSize: number;
}): RadarComposedData;
export declare const defaultRadarProps: {
    readonly activeDot: true;
    readonly angleAxisId: 0;
    readonly animationBegin: 0;
    readonly animationDuration: 1500;
    readonly animationEasing: "ease";
    readonly dot: false;
    readonly hide: false;
    readonly isAnimationActive: "auto";
    readonly label: false;
    readonly legendType: "rect";
    readonly radiusAxisId: 0;
    readonly zIndex: 100;
};
/**
 * @consumes PolarChartContext
 * @provides LabelListContext
 */
export declare function Radar(outsideProps: Props): React.JSX.Element;
export declare namespace Radar {
    var displayName: string;
}
export {};
