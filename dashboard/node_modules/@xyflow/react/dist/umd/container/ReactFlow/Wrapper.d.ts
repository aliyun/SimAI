import { type ReactNode } from 'react';
import type { Node, Edge, FitViewOptions } from '../../types';
import { CoordinateExtent, NodeOrigin, ZIndexMode } from '@xyflow/system';
export declare function Wrapper({ children, nodes, edges, defaultNodes, defaultEdges, width, height, fitView, fitViewOptions, minZoom, maxZoom, nodeOrigin, nodeExtent, zIndexMode, }: {
    children: ReactNode;
    nodes?: Node[];
    edges?: Edge[];
    defaultNodes?: Node[];
    defaultEdges?: Edge[];
    width?: number;
    height?: number;
    fitView?: boolean;
    fitViewOptions?: FitViewOptions;
    minZoom?: number;
    maxZoom?: number;
    nodeOrigin?: NodeOrigin;
    nodeExtent?: CoordinateExtent;
    zIndexMode?: ZIndexMode;
}): import("react/jsx-runtime").JSX.Element;
//# sourceMappingURL=Wrapper.d.ts.map