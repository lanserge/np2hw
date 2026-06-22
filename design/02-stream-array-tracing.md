# 02 — StreamArray: tracing approach


> **Design rationale** (the *why* behind this part). For current usage and the
> implemented API, see [`docs/`](../docs/). The tracing front-end described here
> is implemented as `Traced` in `frontend.py` — the early `StreamArray` / `core.py`
> prototypes have been removed; the design approach they describe lives on in `Traced`.

## Decision: tracing, not AST parsing

We use a custom `StreamArray` class that mimics the NumPy interface and builds an operation graph through Python execution. Same pattern as JAX, PyTorch, CuPy, TVM, Halide, Dask.

### Why tracing wins over AST parsing

- **Less implementation work** — Python's interpreter does the dispatch
- **Standard NumPy code works** via operator overloads + NumPy's `__array_ufunc__` and `__array_function__` protocols
- **Hash-cons happens naturally** in operator methods (no separate cache layer)
- **Shape/dtype propagation is free** — each method computes from inputs
- **User-defined functions just work** — no inlining or call-graph analysis needed
- **Native Python error messages** — tracebacks point to user code
- **Trace-time Python control flow** — `if shape[0] > 1000:` is decided when tracing

### Trade-offs

- Loops unroll at trace time (for fixed sizes); for ML-scale loops we'd need a separate mechanism, but this is fine for ISP where loops are small
- Some NumPy patterns (fancy indexing assignment) are awkward
- Need to handle the trace-time vs runtime distinction carefully in docs

## StreamArray class design

```python
class StreamArray:
    _next_id = 0
    _hash_cons = {}        # global cache: (op, inputs, params) → StreamArray
    _all_nodes = {}        # global registry: id → StreamArray

    def __init__(self, shape, dtype, bitwidth, producer_op, 
                 producer_inputs=(), producer_params=(),
                 row_offset=0, col_offset=0):
        self.id = StreamArray._next_id
        StreamArray._next_id += 1
        self.shape = shape
        self.dtype = dtype
        self.bitwidth = bitwidth
        self.producer_op = producer_op
        self.producer_inputs = producer_inputs
        self.producer_params = producer_params
        self.row_offset = row_offset
        self.col_offset = col_offset
        StreamArray._all_nodes[self.id] = self
    
    @classmethod
    def _build(cls, op, inputs, params=(), bitwidth_out=None, dtype_out=None):
        canonical_inputs = cls._canonicalise(op, inputs)
        key = (op, tuple(i.id for i in canonical_inputs), tuple(params))
        if key in cls._hash_cons:
            return cls._hash_cons[key]                    # reuse
        # ... compute new node properties from op + inputs ...
        node = cls(...)
        cls._hash_cons[key] = node
        return node
    
    # Operator overloads — trigger graph construction
    def __add__(self, other):
        return StreamArray._build('add', (self, self._lift(other)))
    
    def __mul__(self, other):
        return StreamArray._build('multiply', (self, self._lift(other)))
    
    def __matmul__(self, other):
        return StreamArray._build('matmul', (self, self._lift(other)))
    
    # NumPy intercept protocols
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Catches np.add, np.multiply, np.maximum, np.sin, etc.
        ...
    
    def __array_function__(self, func, types, args, kwargs):
        # Catches np.matmul, np.convolve, np.clip, np.sum, etc.
        ...
```

## User-facing API

```python
@hls_compile(
    inputs={
        'img': dict(shape=(1080, 1920), dtype=np.uint8),
        'kernel': dict(shape=(3, 3), dtype=np.int8, is_constant=True),
    },
    outputs={
        'result': dict(shape=(1080, 1920), dtype=np.uint8, mode='saturate'),
    },
)
def my_pipeline(img, kernel):
    # User writes ordinary-looking NumPy
    blurred = np.convolve(img, kernel)
    return np.clip(blurred, 0, 255)

# Then:
verilog = my_pipeline.to_verilog()
result = my_pipeline.simulate(img=test_img, kernel=test_kernel)
graph = my_pipeline.graph()    # for debugging
```

## Canonicalisation for hash-cons

Commutative ops need canonical input ordering so `add(a, b)` and `add(b, a)` hash to the same key:

```python
COMMUTATIVE_OPS = {'add', 'multiply', 'maximum', 'minimum', 'logical_and', 'logical_or'}

@staticmethod
def _canonicalise(op, inputs):
    if op in COMMUTATIVE_OPS:
        return tuple(sorted(inputs, key=lambda x: x.id))
    return tuple(inputs)
```

Constants must be in `params` (hashable), not as separate ids.

## Hashability rules

Everything in the hash key must be hashable. Specifically:
- Convert lists to tuples
- Convert numpy arrays (for coefficient params) to `tuple(arr.flatten().tolist())`
- Convert dicts to `frozenset(d.items())`
- Floats must be exact (no nan in keys; round explicitly if needed)
