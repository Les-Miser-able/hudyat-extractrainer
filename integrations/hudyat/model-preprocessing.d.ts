export function normalizationFor(preprocessing?: {normalization?: string}): string;
export function normalizeFrames(frames: Float32Array[], width: number, height: number): Float32Array[];
export function preprocessSequence(frames: Float32Array[], width: number, height: number, preprocessing?: {normalization?: string}): Float32Array[];
