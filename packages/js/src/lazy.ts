type BinaryOp = "add" | "sub" | "mul" | "div";

export type ElementwiseFn = (
  value: number,
  meta: { index: number; row: number; col: number; shape: Shape }
) => number;

type Shape = {
  rows: number;
  cols: number;
  size: number;
};

type InputNode = {
  kind: "input";
  shape: Shape;
  data: Float64Array;
};

type ScalarNode = {
  kind: "scalar";
  value: number;
};

type MapNode = {
  kind: "map";
  shape: Shape;
  input: ArrayNode;
  fn: ElementwiseFn;
};

type BinaryNode = {
  kind: "binary";
  shape: Shape;
  left: ArrayNode;
  right: ArrayNode | ScalarNode;
  op: BinaryOp;
};

type ReshapeNode = {
  kind: "reshape";
  shape: Shape;
  input: ArrayNode;
};

type TransposeNode = {
  kind: "transpose";
  shape: Shape;
  input: ArrayNode;
};

type ReduceNode = {
  kind: "reduce";
  input: ArrayNode;
  reduceKind: "sum";
  mapFn?: ElementwiseFn;
};

type ArrayNode = InputNode | MapNode | BinaryNode | ReshapeNode | TransposeNode;
type AnyNode = ArrayNode | ScalarNode | ReduceNode;

type EvaluatedArray = {
  data: Float64Array;
  shape: Shape;
};

function makeShape(rows: number, cols: number): Shape {
  if (!Number.isInteger(rows) || !Number.isInteger(cols) || rows <= 0 || cols <= 0) {
    throw new Error(`Invalid shape (${rows} x ${cols}); dimensions must be positive integers`);
  }
  return { rows, cols, size: rows * cols };
}

function ensureSize(shape: Shape, length: number): void {
  if (shape.size !== length) {
    throw new Error(
      `Data length (${length}) does not match shape (${shape.rows} x ${shape.cols})`
    );
  }
}

function isScalarNode(node: AnyNode): node is ScalarNode {
  return node.kind === "scalar";
}

function composeElementwise(first: ElementwiseFn, second?: ElementwiseFn): ElementwiseFn {
  if (!second) {
    return first;
  }
  return (value, meta) => second(first(value, meta), meta);
}

function makeScalarNode(value: number): ScalarNode {
  if (!Number.isFinite(value)) {
    throw new Error("Lazy scalar value must be a finite number");
  }
  return { kind: "scalar", value };
}

function evaluateArray(node: ArrayNode): EvaluatedArray {
  switch (node.kind) {
    case "input":
      return { data: node.data.slice(), shape: node.shape };
    case "map": {
      const evaluated = evaluateArray(node.input);
      const { data, shape } = evaluated;
      const out = new Float64Array(shape.size);
      for (let index = 0; index < shape.size; index += 1) {
        const row = Math.floor(index / shape.cols);
        const col = index - row * shape.cols;
        out[index] = node.fn(data[index], { index, row, col, shape });
      }
      return { data: out, shape };
    }
    case "binary": {
      const leftEval = evaluateArray(node.left);
      const leftData = leftEval.data;
      const shape = leftEval.shape;
      const out = new Float64Array(shape.size);
      if (isScalarNode(node.right)) {
        const scalar = node.right.value;
        for (let index = 0; index < shape.size; index += 1) {
          out[index] = applyBinaryOp(leftData[index], scalar, node.op);
        }
        return { data: out, shape };
      }
      const rightEval = evaluateArray(node.right);
      const rightData = rightEval.data;
      ensureSameShape(shape, rightEval.shape);
      for (let index = 0; index < shape.size; index += 1) {
        out[index] = applyBinaryOp(leftData[index], rightData[index], node.op);
      }
      return { data: out, shape };
    }
    case "reshape": {
      const evaluated = evaluateArray(node.input);
      ensureSize(node.shape, evaluated.data.length);
      return { data: evaluated.data, shape: node.shape };
    }
    case "transpose": {
      const evaluated = evaluateArray(node.input);
      const inputShape = evaluated.shape;
      const { rows, cols } = node.shape;
      if (rows !== inputShape.cols || cols !== inputShape.rows) {
        throw new Error("Transpose shape mismatch after optimisation");
      }
      const input = evaluated.data;
      const out = new Float64Array(node.shape.size);
      for (let r = 0; r < rows; r += 1) {
        for (let c = 0; c < cols; c += 1) {
          out[r * cols + c] = input[c * rows + r];
        }
      }
      return { data: out, shape: node.shape };
    }
    default:
      throw new Error(`Unsupported array node ${(node as { kind: string }).kind}`);
  }
}

function evaluateScalar(node: ScalarNode | ReduceNode): number {
  if (node.kind === "scalar") {
    return node.value;
  }
  if (node.reduceKind !== "sum") {
    throw new Error(`Unsupported reduce kind: ${node.reduceKind}`);
  }
  const { data, shape } = evaluateArray(node.input);
  const mapFn = node.mapFn;
  let total = 0;
  for (let index = 0; index < shape.size; index += 1) {
    const row = Math.floor(index / shape.cols);
    const col = index - row * shape.cols;
    const value = mapFn
      ? mapFn(data[index], { index, row, col, shape })
      : data[index];
    total += value;
  }
  return total;
}

function applyBinaryOp(lhs: number, rhs: number, op: BinaryOp): number {
  switch (op) {
    case "add":
      return lhs + rhs;
    case "sub":
      return lhs - rhs;
    case "mul":
      return lhs * rhs;
    case "div":
      return lhs / rhs;
  }
  const exhaustive: never = op;
  throw new Error(`Unsupported binary op: ${exhaustive}`);
}

function ensureSameShape(a: Shape, b: Shape): void {
  if (a.rows !== b.rows || a.cols !== b.cols) {
    throw new Error(
      `Shape mismatch: (${a.rows} x ${a.cols}) vs (${b.rows} x ${b.cols})`
    );
  }
}

function optimizeArray(node: ArrayNode): ArrayNode {
  switch (node.kind) {
    case "map": {
      const input = optimizeArray(node.input);
      if (input.kind === "map") {
        return {
          kind: "map",
          input: input.input,
          shape: input.shape,
          fn: composeElementwise(input.fn, node.fn),
        };
      }
      return {
        kind: "map",
        input,
        shape: input.shape,
        fn: node.fn,
      };
    }
    case "binary": {
      const left = optimizeArray(node.left);
      const right = optimizeArrayOrScalar(node.right);
      if (isScalarNode(right)) {
        return {
          kind: "binary",
          left,
          right,
          op: node.op,
          shape: left.shape,
        };
      }
      ensureSameShape(left.shape, right.shape);
      return {
        kind: "binary",
        left,
        right,
        op: node.op,
        shape: left.shape,
      };
    }
    case "reshape": {
      const input = optimizeArray(node.input);
      if (input.kind === "reshape") {
        return {
          kind: "reshape",
          input: input.input,
          shape: node.shape,
        };
      }
      if (input.shape.rows === node.shape.rows && input.shape.cols === node.shape.cols) {
        return input;
      }
      return {
        kind: "reshape",
        input,
        shape: node.shape,
      };
    }
    case "transpose": {
      const input = optimizeArray(node.input);
      if (input.kind === "transpose") {
        return input.input;
      }
      return {
        kind: "transpose",
        input,
        shape: node.shape,
      };
    }
    case "input":
      return node;
    default:
      throw new Error(`Unsupported node ${(node as { kind: string }).kind}`);
  }
}

function optimizeScalarNode(node: ScalarNode | ReduceNode): ScalarNode | ReduceNode {
  if (node.kind === "scalar") {
    return node;
  }
  const optimisedInput = optimizeArray(node.input);
  let fusedMap = node.mapFn;
  let source = optimisedInput;
  if (source.kind === "map") {
    fusedMap = composeElementwise(source.fn, fusedMap);
    source = source.input;
  }
  return {
    kind: "reduce",
    input: source,
    reduceKind: node.reduceKind,
    mapFn: fusedMap,
  };
}

function optimizeArrayOrScalar(node: ArrayNode | ScalarNode): ArrayNode | ScalarNode {
  return isScalarNode(node) ? node : optimizeArray(node);
}

export class LazyArray {
  private readonly node: ArrayNode;

  private constructor(node: ArrayNode) {
    this.node = node;
  }

  static fromDense(data: Float64Array | Float32Array | number[], rows: number, cols: number): LazyArray {
    const shape = makeShape(rows, cols);
    const array = data instanceof Float64Array
      ? data.slice()
      : Float64Array.from(data);
    ensureSize(shape, array.length);
    return new LazyArray({
      kind: "input",
      shape,
      data: array,
    });
  }

  static fromConstant(value: number, rows: number, cols: number): LazyArray {
    const shape = makeShape(rows, cols);
    const data = new Float64Array(shape.size).fill(value);
    return new LazyArray({
      kind: "input",
      shape,
      data,
    });
  }

  get shape(): { rows: number; cols: number } {
    return { rows: this.node.shape.rows, cols: this.node.shape.cols };
  }

  map(fn: ElementwiseFn): LazyArray {
    return new LazyArray({
      kind: "map",
      input: this.node,
      shape: this.node.shape,
      fn,
    });
  }

  add(other: number | LazyArray): LazyArray {
    return this.binary("add", other);
  }

  sub(other: number | LazyArray): LazyArray {
    return this.binary("sub", other);
  }

  mul(other: number | LazyArray): LazyArray {
    return this.binary("mul", other);
  }

  div(other: number | LazyArray): LazyArray {
    return this.binary("div", other);
  }

  reshape(rows: number, cols: number): LazyArray {
    const shape = makeShape(rows, cols);
    ensureSize(shape, this.node.shape.size);
    return new LazyArray({
      kind: "reshape",
      input: this.node,
      shape,
    });
  }

  transpose(): LazyArray {
    return new LazyArray({
      kind: "transpose",
      input: this.node,
      shape: makeShape(this.node.shape.cols, this.node.shape.rows),
    });
  }

  reduceSum(): LazyScalar {
    return new LazyScalar({
      kind: "reduce",
      input: this.node,
      reduceKind: "sum",
    });
  }

  evaluate(): { data: Float64Array; rows: number; cols: number } {
    const optimised = optimizeArray(this.node);
    const result = evaluateArray(optimised);
    return {
      data: result.data,
      rows: result.shape.rows,
      cols: result.shape.cols,
    };
  }

  private binary(op: BinaryOp, other: number | LazyArray): LazyArray {
    if (typeof other === "number") {
      return new LazyArray({
        kind: "binary",
        left: this.node,
        right: makeScalarNode(other),
        op,
        shape: this.node.shape,
      });
    }
    ensureSameShape(this.node.shape, other.node.shape);
    return new LazyArray({
      kind: "binary",
      left: this.node,
      right: other.node,
      op,
      shape: this.node.shape,
    });
  }
}

export class LazyScalar {
  private readonly node: ScalarNode | ReduceNode;

  constructor(node: ScalarNode | ReduceNode) {
    this.node = node;
  }

  static fromNumber(value: number): LazyScalar {
    return new LazyScalar(makeScalarNode(value));
  }

  evaluate(): number {
    const optimised = optimizeScalarNode(this.node);
    return evaluateScalar(optimised);
  }
}

export function constant(value: number, rows = 1, cols = 1): LazyArray {
  return LazyArray.fromConstant(value, rows, cols);
}
