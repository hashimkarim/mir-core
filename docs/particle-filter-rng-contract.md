# Coupled historical and portable RNG validation

`mir_core.testing.particle_filter_contract` checks the actual historical NumPy
and portable SplitMix64 trackers against the same frozen contract manifest.
Both random-call tapes and per-frame particle/event traces have immutable
digests. Instrumented runs must match uninstrumented runs, and validation
restores the ambient NumPy state.

Run the shared source gate in the MIR environment:

```bash
PYTHONPATH=. python -m mir_core.testing.particle_filter_contract \
  --output build/particle-filter-contract.json
```

The native repository imports this same checker. It stores binary tapes for
both RNGs and replays them through the production filter source in a dedicated
test build, then checks the actual production generator separately. Native and
shared manifests must match; changing one cannot silently change the contract
used by the other. The workspace `tools/validation/validate_ports.py` reference
phase requires the full native and Python gate.

Python CI verifies the source contract directly. Native CI pins a public Python
checkpoint and adds native replay and distribution checks. This keeps the
contract shared without giving public CI access to the private native code.

Four profiles exercise ordinary, locked-meter, packaged and sustained behavior,
with the sustained profile running 4,096 frames. Historical and portable events
can differ because their RNG algorithms differ. Each must preserve its own
reference behavior and agree with the shared native state updates when supplied
the same random draws. Fault-injection tests deliberately break both generators,
the common state updates, and the manifest link to check that these regressions
are rejected.

Legacy research behavior remains available under `legacy-numpy-global-v1`;
deployment retains `portable-splitmix64-v1`. Held-out musical accuracy and
inference performance remain separate from this compatibility contract.
Baselines are rewritten only by the explicit `--write` maintenance command;
CI never regenerates them.
