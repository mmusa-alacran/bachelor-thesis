# Third-party notices

## neuralpredictors

The following files in `model/` are derived from the neuralpredictors library
(<https://github.com/sinzlab/neuralpredictors>) and remain under its MIT licence:

| File | Upstream source |
|---|---|
| `model/np_point_pooled.py` | `neuralpredictors/layers/readouts/point_pooled.py` |
| `model/np_readout_base.py` | `neuralpredictors/layers/readouts/base.py` |
| `model/np_transfer_learning_core.py` | `neuralpredictors/layers/cores/conv2d.py` (class `TransferLearningCore`) |
| `model/np_measures.py` | `neuralpredictors/measures/modules.py` (classes `Corr`, `PoissonLoss`) |

`NegativeBinomialLoss` in `model/np_measures.py` has no upstream counterpart. It is
original to this work and is covered by the licence in `LICENSE`.

They were copied locally, rather than imported, so that a fix to `TransferLearningCore`
(which probes at 64x64 and can infer the wrong `OutBatchNorm` channel count on some
torchvision versions) could be applied. These local copies are the versions that produced
the results reported in the thesis.

Upstream licence, reproduced in full as required:

```
MIT License

Copyright (c) 2019 Sinz Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
