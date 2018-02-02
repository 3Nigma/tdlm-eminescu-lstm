import { Array1D, Array2D, Array3D, Array4D } from '../../ndarray';
import { KernelNode } from '../tape_types';
export interface Slice1DNode extends KernelNode {
    inputAndArgs: {
        inputs: {
            x: Array1D;
        };
        args: {
            begin: number;
            size: number;
        };
    };
    output: Array1D;
    gradient: (dy: Array1D, y: Array1D) => {
        x: () => Array1D;
    };
}
export interface Slice2DNode extends KernelNode {
    inputAndArgs: {
        inputs: {
            x: Array2D;
        };
        args: {
            begin: [number, number];
            size: [number, number];
        };
    };
    output: Array2D;
    gradient: (dy: Array2D, y: Array2D) => {
        x: () => Array2D;
    };
}
export interface Slice3DNode extends KernelNode {
    inputAndArgs: {
        inputs: {
            x: Array3D;
        };
        args: {
            begin: [number, number, number];
            size: [number, number, number];
        };
    };
    output: Array3D;
    gradient: (dy: Array3D, y: Array3D) => {
        x: () => Array3D;
    };
}
export interface Slice4DNode extends KernelNode {
    inputAndArgs: {
        inputs: {
            x: Array4D;
        };
        args: {
            begin: [number, number, number, number];
            size: [number, number, number, number];
        };
    };
    output: Array4D;
    gradient: (dy: Array4D, y: Array4D) => {
        x: () => Array4D;
    };
}
