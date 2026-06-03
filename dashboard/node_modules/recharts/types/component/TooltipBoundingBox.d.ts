import * as React from 'react';
import { CSSProperties, PureComponent, ReactNode } from 'react';
import { AllowInDimension, AnimationDuration, AnimationTiming, CartesianViewBox, Coordinate, PolarCoordinate } from '../util/types';
import { ElementOffset, SetElementOffset } from '../util/useElementOffset';
export type TooltipBoundingBoxProps = {
    active: boolean;
    allowEscapeViewBox: AllowInDimension;
    animationDuration: AnimationDuration;
    animationEasing: AnimationTiming;
    children: ReactNode;
    coordinate: Coordinate | PolarCoordinate | undefined;
    hasPayload: boolean;
    isAnimationActive: boolean | 'auto';
    offset: number | Coordinate;
    position: Partial<Coordinate> | undefined;
    reverseDirection: AllowInDimension;
    useTranslate3d: boolean;
    viewBox: CartesianViewBox;
    wrapperStyle: CSSProperties;
    lastBoundingBox: ElementOffset;
    innerRef: SetElementOffset;
    hasPortalFromProps: boolean;
};
type State = {
    dismissed: boolean;
    dismissedAtCoordinate: Coordinate;
};
export declare class TooltipBoundingBox extends PureComponent<TooltipBoundingBoxProps, State> {
    state: {
        dismissed: boolean;
        dismissedAtCoordinate: {
            x: number;
            y: number;
        };
    };
    componentDidMount(): void;
    componentWillUnmount(): void;
    componentDidUpdate(): void;
    handleKeyDown: (event: KeyboardEvent) => void;
    render(): React.JSX.Element;
}
export {};
