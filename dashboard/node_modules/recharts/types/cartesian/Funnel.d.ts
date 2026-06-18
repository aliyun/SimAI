import * as React from 'react';
import { Props as TrapezoidProps } from '../shape/Trapezoid';
import { ImplicitLabelListType } from '../component/LabelList';
import { ActiveShape, AnimationDuration, AnimationTiming, CartesianViewBoxRequired, ChartOffsetInternal, Coordinate, DataKey, DataProvider, LegendType, PresentationAttributesAdaptChildEvent, TooltipType, TrapezoidViewBox } from '../util/types';
import { TooltipPayload } from '../state/tooltipSlice';
import { GraphicalItemId } from '../state/graphicalItemsSlice';
export type FunnelTrapezoidItem = TrapezoidProps & TrapezoidViewBox & {
    value?: number | string;
    payload?: any;
    tooltipPosition: Coordinate;
    name: string;
    labelViewBox: TrapezoidViewBox;
    parentViewBox: CartesianViewBoxRequired;
    val: number | ReadonlyArray<number>;
    tooltipPayload: TooltipPayload;
};
/**
 * External props, intended for end users to fill in
 */
interface FunnelProps extends DataProvider {
    /**
     * This component is rendered when this graphical item is activated
     * (could be by mouse hover, touch, keyboard, programmatically).
     */
    activeShape?: ActiveShape<FunnelTrapezoidItem, SVGPathElement>;
    /**
     * Specifies when the animation should begin, the unit of this option is ms.
     * @defaultValue 400
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
    className?: string;
    /**
     * Decides how to extract the value of this Funnel from the data:
     * - `string`: the name of the field in the data object;
     * - `number`: the index of the field in the data;
     * - `function`: a function that receives the data object and returns the value of this Funnel.
     */
    dataKey: DataKey<any>;
    /**
     * Hides the whole graphical element when true.
     *
     * Hiding an element is different from removing it from the chart:
     * Hidden graphical elements are still visible in Legend,
     * and can be included in axis domain calculations,
     * depending on `includeHidden` props of your XAxis/YAxis.
     *
     * @defaultValue false
     */
    hide?: boolean;
    /**
     * Unique identifier of this component.
     * Used as an HTML attribute `id`, and also to identify this element internally.
     *
     * If undefined, Recharts will generate a unique ID automatically.
     */
    id?: string;
    /**
     * If set false, animation of funnel will be disabled.
     * If set "auto", the animation will be disabled in SSR and enabled in browser.
     * @defaultValue auto
     */
    isAnimationActive?: boolean | 'auto';
    label?: ImplicitLabelListType;
    /**
     * @defaultValue triangle
     */
    lastShapeType?: 'triangle' | 'rectangle';
    /**
     * The type of icon in legend.  If set to 'none', no legend item will be rendered.
     * @defaultValue rect
     */
    legendType?: LegendType;
    /**
     * Name represents each sector in the tooltip.
     * This allows you to extract the name from the data:
     *
     * - `string`: the name of the field in the data object;
     * - `number`: the index of the field in the data;
     * - `function`: a function that receives the data object and returns the name.
     *
     * @defaultValue name
     */
    nameKey?: DataKey<any>;
    /**
     * The customized event handler of animation end
     */
    onAnimationEnd?: () => void;
    /**
     * The customized event handler of animation start
     */
    onAnimationStart?: () => void;
    reversed?: boolean;
    /**
     * If set a ReactElement, the shape of funnel can be customized.
     * If set a function, the function will be called to render customized shape.
     */
    shape?: ActiveShape<FunnelTrapezoidItem, SVGPathElement>;
    tooltipType?: TooltipType;
    /**
     * The customized event handler of click on the area in this group
     */
    onClick?: (data: any, index: number, e: React.MouseEvent) => void;
    /**
     * The customized event handler of mousedown on the area in this group
     */
    onMouseDown?: (data: any, index: number, e: React.MouseEvent) => void;
    /**
     * The customized event handler of mouseup on the area in this group
     */
    onMouseUp?: (data: any, index: number, e: React.MouseEvent) => void;
    /**
     * The customized event handler of mousemove on the area in this group
     */
    onMouseMove?: (data: any, index: number, e: React.MouseEvent) => void;
    /**
     * The customized event handler of mouseover on the area in this group
     */
    onMouseOver?: (data: any, index: number, e: React.MouseEvent) => void;
    /**
     * The customized event handler of mouseout on the area in this group
     */
    onMouseOut?: (data: any, index: number, e: React.MouseEvent) => void;
    /**
     * The customized event handler of mouseenter on the area in this group
     */
    onMouseEnter?: (data: any, index: number, e: React.MouseEvent) => void;
    /**
     * The customized event handler of mouseleave on the area in this group
     */
    onMouseLeave?: (data: any, index: number, e: React.MouseEvent) => void;
}
type FunnelSvgProps = Omit<PresentationAttributesAdaptChildEvent<any, SVGPathElement>, 'ref'>;
export type Props = FunnelSvgProps & FunnelProps;
type RealFunnelData = unknown;
export declare const defaultFunnelProps: {
    readonly animationBegin: 400;
    readonly animationDuration: 1500;
    readonly animationEasing: "ease";
    readonly fill: "#808080";
    readonly hide: false;
    readonly isAnimationActive: "auto";
    readonly lastShapeType: "triangle";
    readonly legendType: "rect";
    readonly nameKey: "name";
    readonly reversed: false;
    readonly stroke: "#fff";
};
export declare function computeFunnelTrapezoids({ dataKey, nameKey, displayedData, tooltipType, lastShapeType, reversed, offset, customWidth, graphicalItemId, }: {
    dataKey: Props['dataKey'];
    nameKey: Props['nameKey'];
    offset: ChartOffsetInternal;
    displayedData: ReadonlyArray<RealFunnelData>;
    tooltipType?: TooltipType;
    lastShapeType?: Props['lastShapeType'];
    reversed?: boolean;
    customWidth: number | string | undefined;
    graphicalItemId: GraphicalItemId;
}): ReadonlyArray<FunnelTrapezoidItem>;
/**
 * @consumes CartesianViewBoxContext
 * @provides LabelListContext
 * @provides CellReader
 */
export declare function Funnel(outsideProps: Props): React.JSX.Element;
export declare namespace Funnel {
    var displayName: string;
}
export {};
